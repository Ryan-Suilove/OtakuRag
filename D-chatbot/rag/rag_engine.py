import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 修复 OpenMP 冲突


class RAGEngine:
    def __init__(self, 
                 index_path="faiss_index_v2", 
                 model_name="shibing624/text2vec-base-chinese", #语义匹配而非关键词匹配
                 api_url="http://localhost:1234/v1"):
        
        self.index_path = index_path
        self.model_name = model_name
        self.api_url = api_url
        
        # 初始化组件
        self.embeddings = self._init_embeddings()
        self.vector_db = self._load_vector_db()
        self.llm = self._init_llm()
        self.chain = self._build_chain()

    def _init_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'}
        )

    def _load_vector_db(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"找不到索引文件: {self.index_path}")#欧式距离（L2 Distance）
        return FAISS.load_local(
            self.index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def _init_llm(self):
        return ChatOpenAI(
            base_url=self.api_url,
            api_key="lm-studio",
            temperature=0.3 # 稍微调低一点，减少幻觉
        )

    def _build_chain(self):
        # 定义 Prompt
        prompt = ChatPromptTemplate.from_template("""你是一个叫A同学的，精通二次元知识的，自嘲废宅风的AI助手，正在与用户进行交流。
现在请结合以下提供的番剧资料来回答,禁止编造，如果查不到相关资料则回复不知道。

【已知资料】
{context}

【用户问题】
{question}

A同学的回答：""")

        def format_docs(docs):
            # 可以在这里加入调试信息，看检索到了什么
            print(f"\n[DEBUG] 检索到了 {len(docs)} 条片段")
            for i, d in enumerate(docs):
              print(f"  - 片段{i}: {d.page_content[:50]}...")
            return "\n\n".join(doc.page_content for doc in docs)

        # 构建 LCEL 链
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, query):
        """外部调用的统一接口"""
        return self.chain.invoke(query)