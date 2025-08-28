#加载LlamaIndex存储的code/C3中的向量数据
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
# 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")  
Settings.llm = None  # 显式禁用 LLM，防止自动加载 OpenAI
# 加载存储上下文
persist_path = "./llamaindex_index_store"
if not os.path.exists(persist_path):
    raise ValueError(f"Persist path {persist_path} does not exist. Please run the index creation script first.")
storage_context = StorageContext.from_defaults(persist_dir=persist_path)

# 根据存储上下文加载索引
index=load_index_from_storage(storage_context)
# 检查是否正确加载，并打印加载的索引总条数
print(f"Loaded index with {len(index.docstore.docs)} documents.")
# 构建查询引擎
query_engine = index.as_query_engine() 
query = "LlamaIndex是提供了什么工具？"
# 执行文本相似度查询
nodes=query_engine.retrieve(query)
print("\n相似性检索结果:")
for i, node in enumerate(nodes):
    print(f"{i+1}. {node.text}")
print("\n-----------------------\n")
#执行查询
response = query_engine.query(query)
print(f"\n查询: '{query}'")
print("查询结果:")
print(response)  # 打印查询结果
