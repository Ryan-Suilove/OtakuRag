"""
自动从动漫知识库提取实体关系三元组并存入Neo4j图数据库
"""

import os
import json
from glob import glob
from openai import OpenAI
from neo4j import GraphDatabase

# API配置
API_KEY = "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"
MODEL_ID = "qwen/qwen3.5-plus-02-15"
BASE_URL = "https://openrouter.ai/api/v1"

# Neo4j配置
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 知识库路径（相对于此脚本位置）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(SCRIPT_DIR, "..", "wiki", "anime_knowledge_base")


class KnowledgeGraphBuilder:
    def __init__(self):
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        # 初始化Neo4j连接
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def extract_triplets(self, wiki_content: str, anime_name: str) -> list:
        """
        使用LLM从wiki内容中提取实体-关系-实体三元组
        """
        prompt = f"""你是一个知识图谱构建专家。请从以下动漫《{anime_name}》的资料中提取实体-关系-实体三元组。

要求：
1. 提取所有重要的人物、组织、作品等实体
2. 关系需要明确且有意义，例如：
   - 角色A - 女主角 - 作品名
   - 角色A - 声优 - 声优名
   - 角色A - 恋人 - 角色B
   - 角色A - 朋友 - 角色B
   - 作品 - 制作公司 - 公司名
   - 作品 - 原作 - 作者名
   - 作品 - 类型 - 类型名
   - 角色 - 身份 - 身份描述
3. 尽可能提取完整的关系，包括人物关系、作品信息等
4. 每个三元组一行，格式为：实体1|关系|实体2

动漫资料：
{wiki_content}

请直接输出三元组，每行一个，格式为：实体1|关系|实体2
不要输出其他内容。"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            triplets = []

            for line in result.split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        entity1 = parts[0].strip()
                        relation = parts[1].strip()
                        entity2 = '|'.join(parts[2:]).strip()  # 处理实体2中可能包含|的情况
                        if entity1 and relation and entity2:
                            triplets.append((entity1, relation, entity2))

            return triplets
        except Exception as e:
            print(f"提取三元组失败: {e}")
            return []

    def create_entity_and_relation(self, tx, entity1: str, relation: str, entity2: str):
        """
        在Neo4j中创建实体和关系
        """
        # 创建实体节点（使用MERGE避免重复）
        tx.run("""
            MERGE (e1:Entity {name: $entity1})
            MERGE (e2:Entity {name: $entity2})
            MERGE (e1)-[r:RELATION {type: $relation}]->(e2)
        """, entity1=entity1, entity2=entity2, relation=relation)

    def save_to_neo4j(self, triplets: list):
        """
        将三元组保存到Neo4j
        """
        with self.driver.session() as session:
            for entity1, relation, entity2 in triplets:
                try:
                    session.execute_write(self.create_entity_and_relation, entity1, relation, entity2)
                    print(f"  已保存: {entity1} - {relation} - {entity2}")
                except Exception as e:
                    print(f"  保存失败: {entity1} - {relation} - {entity2}, 错误: {e}")

    def process_all_wiki_files(self):
        """
        处理所有wiki文件
        """
        wiki_files = glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.md"))
        total_files = len(wiki_files)
        print(f"共发现 {total_files} 个wiki文件")

        for idx, wiki_file in enumerate(wiki_files, 1):
            anime_name = os.path.splitext(os.path.basename(wiki_file))[0]
            print(f"\n[{idx}/{total_files}] 处理: {anime_name}")

            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取三元组
                print(f"  正在提取三元组...")
                triplets = self.extract_triplets(content, anime_name)
                print(f"  提取到 {len(triplets)} 个三元组")

                if triplets:
                    # 保存到Neo4j
                    print(f"  正在保存到Neo4j...")
                    self.save_to_neo4j(triplets)
                    print(f"  保存完成")

            except Exception as e:
                print(f"  处理失败: {e}")

        print("\n所有文件处理完成!")

    def process_single_wiki(self, wiki_file: str):
        """
        处理单个wiki文件
        """
        anime_name = os.path.splitext(os.path.basename(wiki_file))[0]
        print(f"处理: {anime_name}")

        with open(wiki_file, 'r', encoding='utf-8') as f:
            content = f.read()

        triplets = self.extract_triplets(content, anime_name)
        print(f"提取到 {len(triplets)} 个三元组")

        if triplets:
            self.save_to_neo4j(triplets)

        return triplets


def main():
    builder = KnowledgeGraphBuilder()
    try:
        builder.process_all_wiki_files()
    finally:
        builder.close()


if __name__ == "__main__":
    main()