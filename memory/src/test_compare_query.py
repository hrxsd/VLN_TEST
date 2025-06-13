import json
from typing import Dict

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim


def transform_query(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query}"


def pooling(outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1
        ) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()


def main(json_path: str):
    # Step 1: 加载 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Step 2: 用户输入查询语句
    query = input("请输入一句话：").strip()
    query_prompted = transform_query(query)

    # Step 3: 加载模型
    model_id = 'mixedbread-ai/mxbai-embed-large-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).cuda()

    # Step 4: 编码用户查询
    inputs = tokenizer(query_prompted, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
    query_embedding = pooling(outputs, inputs, 'cls')  # shape: (1, dim)
    query_tensor = torch.from_numpy(query_embedding.astype(np.float32))

    # Step 5: 计算相似度
    results = []
    for idx, item in enumerate(data):
        caption = item['caption']
        embedding = np.array(item['embedding'], dtype=np.float32)
        time = item['time']
        position = item['position']

        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)  # shape: (1, dim)
        sim = cos_sim(query_tensor, embedding_tensor)

        results.append({
            'index': idx,
            'caption': caption,
            'time': time,
            'position': position,
            'similarity': sim.item()
        })

    # Step 6: 排序并输出前5条
    results.sort(key=lambda x: x['similarity'], reverse=True)
    print("\n=== 相似度最高的前5条结果 ===\n")
    for i, r in enumerate(results[:5]):
        print(f"Top {i+1}（原始索引: {r['index']})")
        print(f"Caption: {r['caption']}")
        print(f"Time: {r['time']}")
        print(f"Position: {r['position']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print("-" * 50)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python compare_query.py your_data.json")
    else:
        main(sys.argv[1])
