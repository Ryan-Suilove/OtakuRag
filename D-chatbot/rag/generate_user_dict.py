"""
从 Neo4j 图数据库提取实体名并生成 jieba 自定义词典
用于防止分词器切碎动漫专有名词
"""

import os
from neo4j import GraphDatabase

# Neo4j 配置
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 输出词典文件路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "user_dict.txt")


class UserDictGenerator:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def get_all_entities(self):
        """
        从 Neo4j 提取所有实体名称
        """
        query = """
        MATCH (e:Entity)
        RETURN DISTINCT e.name AS name
        ORDER BY e.name
        """
        entities = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                name = record["name"]
                if name and name.strip():
                    entities.append(name.strip())
        return entities

    def generate_user_dict(self, output_path: str = OUTPUT_FILE):
        """
        生成 jieba 格式的自定义词典

        jieba 词典格式: 词语 词频 词性
        - 词频越高，越不容易被切分
        - 词性可选，这里统一标记为 'nz' (其他专名)
        """
        print("正在从 Neo4j 提取实体...")
        entities = self.get_all_entities()
        print(f"共提取到 {len(entities)} 个实体")

        if not entities:
            print("警告: 未提取到任何实体，请检查 Neo4j 数据库")
            return

        # 按词语长度降序排序，优先处理长词
        entities.sort(key=len, reverse=True)

        # 写入词典文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for entity in entities:
                # 格式: 词语 词频 词性
                # 词频设为较大值，确保优先切分
                # 词性 'nz' 表示其他专名
                f.write(f"{entity} 1000 nz\n")

        print(f"词典已生成: {output_path}")
        print(f"共写入 {len(entities)} 个词条")

    def test_segmentation(self, test_text: str = None):
        """
        测试分词效果
        """
        import jieba

        # 加载自定义词典
        jieba.load_userdict(OUTPUT_FILE)

        if test_text is None:
            test_text = "凉宫春日是凉宫春日的忧郁中的女主角"

        print(f"\n测试分词: {test_text}")
        result = jieba.lcut(test_text)
        print(f"分词结果: {' / '.join(result)}")


def main():
    generator = UserDictGenerator()
    try:
        generator.generate_user_dict()

        # 可选: 测试分词效果
        print("\n是否测试分词效果? (需要安装 jieba)")
        print("运行: python -c \"from generate_user_dict import UserDictGenerator; g=UserDictGenerator(); g.test_segmentation()\"")
    finally:
        generator.close()


if __name__ == "__main__":
    main()