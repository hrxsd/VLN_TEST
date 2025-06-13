import json
import math
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim
from modelscope import AutoModelForCausalLM, AutoTokenizer as QwenTokenizer
import re


class QwenChatbot:
    """Qwen3本地聊天机器人"""
    def __init__(self, model_name="Qwen/Qwen3-4B"):
        self.tokenizer = QwenTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input, max_tokens=512):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )[0][len(inputs.input_ids[0]):].tolist()
        
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()

    def clear_history(self):
        """清空对话历史"""
        self.history = []


class VectorDatabaseRetrieval:
    def __init__(self, json_path: str, qwen_model_name: str = "Qwen/Qwen3-4B"):
        """
        初始化向量数据库检索系统
        
        Args:
            json_path: 向量数据库JSON文件路径
            qwen_model_name: Qwen模型名称
        """
        self.json_path = json_path
        self.data = self._load_data()
        
        # 初始化Qwen聊天机器人
        print("正在加载Qwen3模型...")
        self.chatbot = QwenChatbot(qwen_model_name)
        
        # 初始化文本嵌入模型
        print("正在加载文本嵌入模型...")
        self.model_id = 'mixedbread-ai/mxbai-embed-large-v1'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        
        # 如果有GPU可用，使用GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("使用GPU加速")
        else:
            print("使用CPU运行")
    
    def _load_data(self) -> List[Dict]:
        """加载向量数据库数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def transform_query(self, query: str) -> str:
        """转换查询语句以适应嵌入模型"""
        return f"Represent this sentence for searching relevant passages: {query}"
    
    def pooling(self, outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
        """池化操作"""
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1
            ) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()
    
    def fl(self, object_query: str) -> List[Dict]:
        """
        文本检索函数 (Text Lookup)
        
        Args:
            object_query: 要检索的关键词或短语
            
        Returns:
            List[Dict]: Top-5最相关的记忆片段
        """
        query_prompted = self.transform_query(object_query)
        
        # 编码查询
        inputs = self.tokenizer(query_prompted, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
        query_embedding = self.pooling(outputs, inputs, 'cls')
        query_tensor = torch.from_numpy(query_embedding.astype(np.float32))
        
        # 计算相似度
        results = []
        for idx, item in enumerate(self.data):
            embedding = np.array(item['embedding'], dtype=np.float32)
            embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)
            sim = cos_sim(query_tensor, embedding_tensor)
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'similarity': sim.item()
            })
        
        # 排序并返回Top-5
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:5]
    
    def fp(self, position: Tuple[float, float, float]) -> List[Dict]:
        """
        空间位置检索函数 (Position Lookup)
        
        Args:
            position: 目标位置坐标 (x, y, w)
            
        Returns:
            List[Dict]: Top-5最近的记忆片段
        """
        x_target, y_target, w_target = position
        
        results = []
        for idx, item in enumerate(self.data):
            x, y, w = item['position']
            # 计算欧几里得距离（只考虑x,y坐标）
            distance = math.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'distance': distance
            })
        
        # 按距离排序并返回Top-5
        results.sort(key=lambda x: x['distance'])
        return results[:5]
    
    def ft(self, time_query: str) -> List[Dict]:
        """
        时间检索函数 (Time Lookup)
        
        Args:
            time_query: 目标时间字符串 "YYYY/MM/DD HH:MM:SS"
            
        Returns:
            List[Dict]: Top-5时间最接近的记忆片段
        """
        target_time = datetime.strptime(time_query, "%Y/%m/%d %H:%M:%S")
        
        results = []
        for idx, item in enumerate(self.data):
            item_time = datetime.strptime(item['time'], "%Y/%m/%d %H:%M:%S")
            time_diff = abs((target_time - item_time).total_seconds())
            
            results.append({
                'index': idx,
                'caption': item['caption'],
                'time': item['time'],
                'position': item['position'],
                'time_diff': time_diff
            })
        
        # 按时间差排序并返回Top-5
        results.sort(key=lambda x: x['time_diff'])
        return results[:5]
    
    def analyze_user_intent(self, user_input: str) -> str:
        """
        使用Qwen3模型分析用户意图并构造检索请求
        
        Args:
            user_input: 用户输入的自然语言指令
            
        Returns:
            str: 包含函数调用的响应
        """
        system_prompt = """你是一个智能机器人的记忆检索助手。用户会用自然语言向你提问，你需要分析用户的意图，并构造相应的检索函数调用。

你有三个检索函数可以使用：
1. fl(object) - 文本检索：基于关键词或描述检索语义相关的记忆
2. fp((x,y,w)) - 位置检索：检索指定位置附近的记忆  
3. ft("YYYY/MM/DD HH:MM:SS") - 时间检索：检索指定时间点附近的记忆

分析用户问题，确定需要调用哪个函数，然后输出函数调用。

示例：
用户："我在哪里见过自动售货机？"
分析：用户想找与自动售货机相关的记忆
输出：fl("vending machine")

用户："昨天下午3点我在做什么？"  
分析：用户想查询特定时间的记忆
输出：ft("2025/02/24 15:00:00")

用户："带我去咖啡厅"
分析：用户想要导航到咖啡厅，需要先找到咖啡厅的位置
输出：fl("coffee shop")

请只输出函数调用，不要输出其他内容。用户问题：""" + user_input

        try:
            # 清空历史记录，确保每次分析都是独立的
            self.chatbot.clear_history()
            response = self.chatbot.generate_response(system_prompt, max_tokens=100)
            
            # 提取函数调用部分
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('fl(') or line.startswith('fp(') or line.startswith('ft('):
                    return line
            
            # 如果没有找到标准格式，尝试从响应中提取
            if 'fl(' in response:
                match = re.search(r'fl\([^)]+\)', response)
                if match:
                    return match.group(0)
            elif 'fp(' in response:
                match = re.search(r'fp\([^)]+\)', response)
                if match:
                    return match.group(0)
            elif 'ft(' in response:
                match = re.search(r'ft\([^)]+\)', response)
                if match:
                    return match.group(0)
            
            # 默认使用文本检索
            return f'fl("{user_input}")'
            
        except Exception as e:
            print(f"Qwen模型调用错误: {e}")
            return f'fl("{user_input}")'  # 默认使用文本检索
    
    def execute_function_call(self, function_call: str) -> List[Dict]:
        """
        执行函数调用
        
        Args:
            function_call: 函数调用字符串
            
        Returns:
            List[Dict]: 检索结果
        """
        try:
            # 解析函数调用
            if function_call.startswith("fl("):
                # 提取fl函数的参数
                match = re.search(r'fl\(["\']([^"\']+)["\']\)', function_call)
                if match:
                    query = match.group(1)
                    return self.fl(query)
            
            elif function_call.startswith("fp("):
                # 提取fp函数的参数
                match = re.search(r'fp\(\(([^)]+)\)\)', function_call)
                if match:
                    coords = match.group(1).split(',')
                    x, y, w = map(float, [c.strip() for c in coords])
                    return self.fp((x, y, w))
            
            elif function_call.startswith("ft("):
                # 提取ft函数的参数
                match = re.search(r'ft\(["\']([^"\']+)["\']\)', function_call)
                if match:
                    time_str = match.group(1)
                    return self.ft(time_str)
            
            print(f"无法解析函数调用: {function_call}")
            return []
            
        except Exception as e:
            print(f"执行函数调用错误: {e}")
            return []
    
    def generate_final_answer(self, user_input: str, retrieved_data: List[Dict]) -> Dict:
        """
        基于检索到的数据生成最终答案
        
        Args:
            user_input: 用户原始输入
            retrieved_data: 检索到的记忆片段
            
        Returns:
            Dict: 最终答案JSON对象
        """
        # 构造上下文
        context = "检索到的相关记忆片段：\n"
        for i, item in enumerate(retrieved_data):
            context += f"记忆{i+1}:\n"
            context += f"- 描述: {item['caption']}\n"
            context += f"- 时间: {item['time']}\n"
            context += f"- 位置: {item['position']}\n\n"
        
        system_prompt = f"""你是一个智能机器人助手。用户向你提问，我已经从记忆数据库中检索了相关信息。请根据这些信息回答用户的问题。

重要要求：
1. 如果用户的问题是关于导航（如"带我去..."、"怎么到..."、"...在哪里"等），你需要在答案中包含position字段
2. 如果不是导航问题，只包含text和time字段  
3. 回答要自然、有用，基于检索到的记忆信息

请严格按照以下JSON格式输出：

导航问题格式：
{{"text": "具体的回答内容", "position": [x, y, w], "time": "YYYY/MM/DD HH:MM:SS"}}

非导航问题格式：
{{"text": "具体的回答内容", "time": "YYYY/MM/DD HH:MM:SS"}}

只输出JSON，不要输出其他内容。

用户问题：{user_input}

{context}"""

        try:
            # 清空历史记录
            self.chatbot.clear_history()
            response = self.chatbot.generate_response(system_prompt, max_tokens=300)
            
            # 尝试从响应中提取JSON
            json_match = re.search(r'\{[^}]*"text"[^}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # 如果无法提取JSON，构造默认响应
            is_navigation = any(keyword in user_input.lower() for keyword in 
                              ['带我去', '怎么到', '在哪里', '在哪儿', '去', '找到', '位置'])
            
            current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            
            if is_navigation and retrieved_data:
                # 使用第一个检索结果的位置
                return {
                    "text": f"根据记忆，我找到了相关位置信息：{retrieved_data[0]['caption']}",
                    "position": retrieved_data[0]['position'],
                    "time": current_time
                }
            else:
                return {
                    "text": f"根据检索到的记忆信息，{response}",
                    "time": current_time
                }
                
        except Exception as e:
            print(f"生成最终答案错误: {e}")
            return {
                "text": "抱歉，我无法处理您的请求。",
                "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            }
    
    def query(self, user_input: str) -> Dict:
        """
        主查询接口
        
        Args:
            user_input: 用户自然语言输入
            
        Returns:
            Dict: 最终答案JSON对象
        """
        print(f"用户输入: {user_input}")
        
        # 第1步：分析用户意图，构造检索请求
        print("正在分析用户意图...")
        function_call = self.analyze_user_intent(user_input)
        print(f"构造的函数调用: {function_call}")
        
        # 第2步：执行检索
        print("正在执行检索...")
        retrieved_data = self.execute_function_call(function_call)
        print(f"检索到 {len(retrieved_data)} 条相关记忆")
        
        # 第3步：生成最终答案
        print("正在生成最终答案...")
        final_answer = self.generate_final_answer(user_input, retrieved_data)
        
        return final_answer


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python vector_retrieval.py <数据库JSON文件路径> [Qwen模型名称]")
        print("默认模型: Qwen/Qwen3-4B")
        return
    
    json_path = sys.argv[1]
    qwen_model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-4B"
    
    # 初始化检索系统
    print("=== 初始化向量数据库检索系统 ===")
    retrieval_system = VectorDatabaseRetrieval(json_path, qwen_model)
    
    print("\n=== 向量数据库检索系统启动 ===")
    print("输入 'quit' 或 'exit' 退出系统\n")
    
    while True:
        try:
            user_input = input("请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("系统已退出")
                break
            
            if not user_input:
                continue
            
            # 执行查询
            result = retrieval_system.query(user_input)
            
            # 输出结果
            print("\n=== 查询结果 ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n系统已退出")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()