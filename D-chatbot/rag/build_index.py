import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#本脚本用于构建番剧知识库的向量索引，供后续 RAG 模块使用
# 配置
DATA_PATH = r"C:\Users\lty20\PycharmProjects\RyanProjects\D-chatbot\wiki\anime_knowledge_base"
SAVE_PATH = "faiss_index"
MODEL_NAME = "shibing624/text2vec-base-chinese"

def build_vector_store():
    # 1. 加载所有 Markdown 文档
    print(f"正在加载文档: {DATA_PATH} ...")
    loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    print(f"成功加载 {len(documents)} 部番剧资料")

    # 2. 文本切分
    # 针对番剧资料，500字一个切片，重叠50字以防断句
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(texts)} 个文本段")

    # 3. 初始化 Embedding 模型 (强制在 CPU 运行以节省显存给 LM Studio)
    print("正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # 4. 创建向量库并保存
    print("正在生成向量并构建索引 (这可能需要几分钟)...")
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(SAVE_PATH)
    print(f"恭喜！向量库已成功保存至: {SAVE_PATH}")

if __name__ == "__main__":
    build_vector_store()