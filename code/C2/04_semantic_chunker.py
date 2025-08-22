
# HuggingFaceEmbeddings 类在 LangChain 0.2.2 版本中已被弃用，并将在 1.0 版本中完全移除。
#具体来说，这个类的功能已经被迁移到了一个独立的 langchain-huggingface 包中，这是 LangChain 团队为了拆分功能、减小核心包体积而做的调整。
#应该替换为：
#pip install -U langchain-huggingface
#from langchain_huggingface import HuggingFaceEmbeddings
import os
## os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# 你可以在 HuggingFace 的模型库 https://huggingface.co/models 搜索你需要的模型，
# model_name 参数就是模型页面的 "Repository name"，比如 "BAAI/bge-small-zh-v1.5"
# 直接复制页面顶部的模型名称字符串即可。
embeddings = HuggingFaceEmbeddings(
    model_name="richinfoai/ritrieve_zh_v1",
    #model_name="BAAI/bge-small-zh-v1.5"
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
)

loader = TextLoader("../../data/C2/txt/蜂医.txt")
documents = loader.load()

docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
#for i, chunk in enumerate(docs[:2]):
for i, chunk in enumerate(docs):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')


#运行结果，这篇文章只分出来两个块，对格式化有层级的md文档效果不好
