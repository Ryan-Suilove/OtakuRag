import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DATA_PATH = r"C:\Users\lty20\PycharmProjects\RyanProjects\D-chatbot\wiki\anime_knowledge_base"
SAVE_PATH = "faiss_index_v2"
MODEL_NAME = "shibing624/text2vec-base-chinese"

def build_vector_store():
    # 1. 初始化相关组件
    # 稍微增加 chunk_size 因为我们要注入前缀，增加 overlap 确保实体词不被切断
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    
    print("正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    all_processed_chunks = []

    # 2. 遍历文件夹，实现“读一个，切一个，加一个前缀”
    print(f"正在逐个处理文档: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print(f"错误：路径不存在 {DATA_PATH}")
        return

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".md")]
    
    for file_name in files:
        file_path = os.path.join(DATA_PATH, file_name)
        anime_title = file_name.replace(".md", "") # 提取番剧名作为前缀
        
        try:
            # 加载单个文件
            loader = TextLoader(file_path, encoding='utf-8')
            single_document = loader.load()[0] # TextLoader 返回的是列表，取第一个
            
            # 独立切分该文档
            chunks = text_splitter.split_text(single_document.page_content)
            
            # 为该文档的所有切片打上“身份标签”
            for chunk in chunks:
                # 核心修改：在文本内容最前面强行加上元数据前缀
                enriched_content = f"【资料所属动漫：{anime_title}】\n{chunk}"
                
                # 封装回 Document 对象
                new_doc = Document(
                    page_content=enriched_content,
                    metadata={"source": anime_title} # 同时在 metadata 存一份备份
                )
                all_processed_chunks.append(new_doc)
            
            print(f"  - 已处理: {anime_title} (切分为 {len(chunks)} 段)")
            
        except Exception as e:
            print(f"  - 处理文件 {file_name} 时出错: {e}")

    if not all_processed_chunks:
        print("未发现有效文本段，取消索引构建。")
        return

    print(f"\n全部处理完成！总计获得 {len(all_processed_chunks)} 个带标签文本段")

    # 3. 创建向量库并保存
    print("正在生成向量并构建索引 (这可能需要几分钟)...")
    vector_db = FAISS.from_documents(all_processed_chunks, embeddings)
    vector_db.save_local(SAVE_PATH)
    print(f"恭喜！向量库已成功保存至: {SAVE_PATH}")

if __name__ == "__main__":
    build_vector_store()