"""
RAG Engine V2 - 双路并行检索策略
同时利用 FAISS 向量库和 Neo4j 知识图谱来解答用户问答
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import jieba
from neo4j import GraphDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RAGEngineV2:
    def __init__(self,
                 index_path="faiss_index_v2",
                 model_name="shibing624/text2vec-base-chinese",
                 api_url="http://localhost:1234/v1",
                 user_dict_path="user_dict.txt",
                 neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_user="neo4j",
                 neo4j_password="12345678"):

        self.index_path = index_path
        self.model_name = model_name
        self.api_url = api_url
        self.user_dict_path = user_dict_path

        # Neo4j 配置
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # 初始化各组件
        self._init_jieba()
        self.embeddings = self._init_embeddings()
        self.vector_db = self._load_vector_db()
        self.neo4j_driver = self._init_neo4j()
        self.llm = self._init_llm()

    def _init_jieba(self):
        """初始化 jieba 分词，加载用户词典"""
        # 获取用户词典的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dict_path = os.path.join(script_dir, self.user_dict_path)

        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
            print(f"[INFO] 已加载用户词典: {dict_path}")
        else:
            print(f"[WARNING] 用户词典不存在: {dict_path}")

    def _init_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'}
        )

    def _load_vector_db(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_index_path = os.path.join(script_dir, self.index_path)

        if not os.path.exists(full_index_path):
            raise FileNotFoundError(f"找不到索引文件: {full_index_path}")
        return FAISS.load_local(
            full_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _init_neo4j(self):
        """初始化 Neo4j 连接"""
        try:
            driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            # 测试连接
            with driver.session() as session:
                session.run("RETURN 1")
            print(f"[INFO] Neo4j 连接成功: {self.neo4j_uri}")
            return driver
        except Exception as e:
            print(f"[WARNING] Neo4j 连接失败: {e}")
            return None

    def _init_llm(self):
        return ChatOpenAI(
            base_url=self.api_url,
            api_key="lm-studio",
            temperature=0.3
        )

    def segment_query(self, query: str) -> list:
        """使用 jieba 对查询进行分词"""
        words = list(jieba.cut(query))
        # 过滤停用词和单字，保留有意义的关键词
        keywords = [w for w in words if len(w) > 1 and w.strip()]
        print(f"[DEBUG] 分词结果: {keywords}")
        return keywords

    def search_neo4j(self, keywords: list) -> str:
        """在 Neo4j 知识图谱中搜索相关实体和关系"""
        if not self.neo4j_driver:
            return ""

        results = []
        seen_entities = set()

        with self.neo4j_driver.session() as session:
            for keyword in keywords:
                # 搜索实体名称匹配的节点
                query = """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $keyword
                MATCH (e)-[r:RELATION]-(related)
                RETURN e.name as entity, r.type as relation, related.name as related_entity
                LIMIT 10
                """
                try:
                    result = session.run(query, keyword=keyword)
                    for record in result:
                        entity = record["entity"]
                        relation = record["relation"]
                        related_entity = record["related_entity"]

                        key = f"{entity}-{relation}-{related_entity}"
                        if key not in seen_entities:
                            seen_entities.add(key)
                            results.append(f"{entity} {relation} {related_entity}")
                except Exception as e:
                    print(f"[DEBUG] Neo4j 查询错误: {e}")

        neo4j_context = "\n".join(results) if results else ""
        print(f"[DEBUG] Neo4j 检索到 {len(results)} 条关系")
        return neo4j_context

    def filter_core_keywords(self, keywords: list) -> list:
        """过滤核心关键词，用于向量检索"""
        # 过滤常见的疑问词和停用词
        stop_words = {'什么', '怎么', '如何', '为什么', '哪里', '谁', '哪个',
                      '吗', '呢', '吧', '的', '了', '是', '有', '在', '和',
                      '与', '或', '但', '如果', '虽然', '因为', '所以', '可以',
                      '能', '会', '应该', '需要', '请', '帮我', '告诉我', '问'}

        core_keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 1]
        print(f"[DEBUG] 核心关键词: {core_keywords}")
        return core_keywords

    def search_faiss(self, keywords: list, k: int = 3) -> str:
        """在 FAISS 向量库中搜索相关文档"""
        # 用关键词组合成查询语句
        query_text = " ".join(keywords)

        retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query_text)

        print(f"[DEBUG] FAISS 检索到 {len(docs)} 条片段")
        for i, doc in enumerate(docs):
            print(f"  - 片段{i}: {doc.page_content[:50]}...")

        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, query: str) -> str:
        """双路并行检索并生成回答"""
        # 1. 使用 jieba 分词
        keywords = self.segment_query(query)

        # 2. 过滤核心关键词
        core_keywords = self.filter_core_keywords(keywords)

        # 3. 并行检索（这里顺序执行，实际可改为并发）
        # Neo4j 检索（使用全部分词结果）
        neo4j_context = self.search_neo4j(keywords)

        # FAISS 检索（使用过滤后的核心关键词）
        faiss_context = self.search_faiss(core_keywords) if core_keywords else self.search_faiss(keywords)

        # 4. 合并上下文
        combined_context = self._merge_contexts(neo4j_context, faiss_context)

        # 5. 生成回答
        response = self._generate_response(query, combined_context)

        return response

    def _merge_contexts(self, neo4j_context: str, faiss_context: str) -> str:
        """合并两个知识库的检索结果"""
        parts = []

        if neo4j_context.strip():
            parts.append("【知识图谱信息】\n" + neo4j_context)

        if faiss_context.strip():
            parts.append("【文档资料】\n" + faiss_context)

        return "\n\n".join(parts) if parts else "未找到相关资料"

    def _generate_response(self, query: str, context: str) -> str:
        """使用 LLM 生成回答"""
        prompt = ChatPromptTemplate.from_template("""你是一个叫A同学的，精通二次元知识的，自嘲废宅风的AI助手，正在与用户进行交流。
现在请结合以下提供的知识图谱信息和文档资料来回答，禁止编造，如果查不到相关资料则回复不知道。

{context}

【用户问题】
{question}

A同学的回答：""")

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    def close(self):
        """关闭资源连接"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def __del__(self):
        self.close()


# 测试入口
if __name__ == "__main__":
    engine = RAGEngineV2()

    test_questions = [
        "鲁迪乌斯是谁？",
        "辉夜大小姐想让我告白的作者是谁？",
        "进击的巨人有什么主要角色？"
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"问题: {q}")
        print(f"{'='*50}")
        answer = engine.invoke(q)
        print(f"\n回答: {answer}")

    engine.close()