import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

# 1. 指定嵌入维度
dimensions = 512

# 2. 加载模型
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

# 3. 输入文本
query = "A man is eating a piece of bread"
docs = [
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

# 4. 编码文本为向量（Embedding）
query_embedding = model.encode(query, prompt_name="query")
docs_embeddings = model.encode(docs)

# 5. 打印向量（可选）
print("Query embedding shape:", query_embedding.shape)
print("First doc embedding shape:", docs_embeddings[0].shape)
print("Query embedding:", query_embedding)
print("First doc embedding:", docs_embeddings[0])
