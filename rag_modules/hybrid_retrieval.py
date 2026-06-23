"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合图结构检索和向量检索，使用Round-robin轮询策略
"""

import json
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'low' or 'high'
    metadata: Dict[str, Any]

class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 关键词提取和匹配
    3. 图结构+向量检索结合
    4. 一跳邻居扩展
    5. Round-robin轮询合并策略
    """
    
    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None
        self.parent_chunk_cache = {}  # parent_chunk_id → parent文本（Parent-Child策略用）
        
        # 图索引模块
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        
    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri, 
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # 构建 Parent 块缓存（Parent-Child 策略用）
        self._build_parent_chunk_cache(chunks)
        
        # 初始化BM25检索器（只用 Child 块，过滤掉 Parent 块）
        child_chunks = [c for c in chunks if c.metadata.get("doc_type") != "parent"]
        if child_chunks:
            self.bm25_retriever = BM25Retriever.from_documents(child_chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(child_chunks)}")
        
        # 初始化图索引
        self._build_graph_index()
        
    def _build_graph_index(self):
        """构建图索引"""
        if self.graph_indexed:
            return
            
        logger.info("开始构建图索引...")
        
        try:
            # 获取图数据
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_steps = self.data_module.cooking_steps
            
            # 创建实体键值对
            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_steps)
            
            # 创建关系键值对（这里需要从Neo4j获取关系数据）
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            
            # 去重优化
            self.graph_indexing.deduplicate_entities_and_relations()
            
            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")
            
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _build_parent_chunk_cache(self, chunks: List[Document]):
        """
        构建 Parent 块缓存：从所有 chunks 中提取 Parent 块，
        以 parent_chunk_id 为 key 存入内存字典。
        检索命中 Child 时，通过 parent_id 从此缓存取 Parent 文本返回。
        """
        self.parent_chunk_cache = {}
        for chunk in chunks:
            if chunk.metadata.get("doc_type") == "parent":
                parent_id = chunk.metadata.get("chunk_id", "")
                if parent_id:
                    self.parent_chunk_cache[parent_id] = chunk.page_content

        if self.parent_chunk_cache:
            logger.info(f"Parent 块缓存构建完成，共 {len(self.parent_chunk_cache)} 个 Parent 块")
        else:
            logger.debug("未检测到 Parent 块，Parent-Child 扩展未启用")

    def _replace_with_parent(self, doc: Document) -> Document:
        """
        如果文档是 Child 块且有对应的 Parent 缓存，
        用 Parent 文本替换 Child 文本返回（上下文更完整）。
        如果没有 Parent 缓存或不是 Child 块，原样返回。
        """
        parent_id = doc.metadata.get("parent_id", "")
        if parent_id and parent_id in self.parent_chunk_cache:
            parent_text = self.parent_chunk_cache[parent_id]
            doc.page_content = parent_text
            doc.metadata["expanded_from_child"] = True
            logger.debug(f"Child 块已扩展为 Parent 块: {parent_id}")
        return doc

    def bm25_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        BM25 全文检索：基于关键词统计的精确匹配，
        作为向量检索和图检索的兜底补充。
        """
        if not self.bm25_retriever:
            logger.warning("BM25 检索器未初始化，跳过 BM25 检索")
            return []

        try:
            docs = self.bm25_retriever.invoke(query)
            # 对每个结果尝试扩展为 Parent 块
            result_docs = []
            for doc in docs[:top_k]:
                doc = self._replace_with_parent(doc)
                doc.metadata["search_method"] = "bm25"
                doc.metadata["search_type"] = "bm25"
                result_docs.append(doc)

            logger.info(f"BM25 检索完成，返回 {len(result_docs)} 个文档")
            return result_docs

        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []
            
    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从Neo4j图中提取关系"""
        relationships = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 1000
                """
                result = session.run(query)
                
                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))
                    
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
            
        return relationships
            
    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """
        提取查询关键词：实体级 + 主题级
        """
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜" 
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])
            
            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级方案：简单的关键词分割
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords
    
    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        实体级检索：专注于具体实体和关系
        使用图索引的键值对结构进行检索
        """
        results = []
        
        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)
            
            for entity in entities:
                # 获取邻居信息
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=2)
                
                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"
                
                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))
        
        # 2. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 3. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j补充检索"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('recipe_fulltext_index', keyword + '*') 
                YIELD node, score
                WHERE node:Recipe
                RETURN 
                    node.nodeId as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"菜品: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")
            
        return results
    
    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        主题级检索：专注于广泛主题和概念
        使用图索引的关系键值对结构进行主题检索
        """
        results = []
        
        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)
            
            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                
                if source_entity and target_entity:
                    # 构建丰富的主题内容
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关菜品: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]
                    
                    # 添加源实体的详细信息
                    if source_entity.entity_type == "Recipe":
                        newline = '\n'
                        content_parts.append(f"菜品详情: {source_entity.value_content.split(newline)[0]}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以主要实体为ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))
        
        # 2. 使用实体的分类信息进行主题检索
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type == "Recipe":
                    # 构建分类主题内容
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"
                        }
                    ))
        
        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j主题级检索补充"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (r:Recipe)
                WHERE r.category CONTAINS keyword 
                   OR r.cuisineType CONTAINS keyword
                   OR r.tags CONTAINS keyword
                WITH r, keyword
                OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                WITH r, keyword, collect(i.name)[0..3] as ingredients
                RETURN 
                    r.nodeId as node_id,
                    r.name as name,
                    r.category as category,
                    r.cuisineType as cuisine_type,
                    r.difficulty as difficulty,
                    ingredients,
                    keyword as matched_keyword
                ORDER BY r.difficulty ASC, r.name
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    content_parts.append(f"菜品: {record['name']}")
                    
                    if record["category"]:
                        content_parts.append(f"分类: {record['category']}")
                    if record["cuisine_type"]:
                        content_parts.append(f"菜系: {record['cuisine_type']}")
                    if record["difficulty"]:
                        content_parts.append(f"难度: {record['difficulty']}")
                    
                    if record["ingredients"]:
                        ingredients_str = ', '.join(record["ingredients"][:3])
                        content_parts.append(f"主要食材: {ingredients_str}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "cuisine_type": record["cuisine_type"],
                            "difficulty": record["difficulty"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")
            
        return results
        
    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """
        双层检索：结合实体级和主题级检索
        """
        logger.info(f"开始双层检索: {query}")
        
        # 1. 提取关键词
        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        
        # 2. 执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)
        
        # 3. 结果合并和排序
        all_results = entity_results + topic_results
        
        # 4. 去重和重排序
        seen_nodes = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)
        
        # 5. 转换为Document格式
        documents = []
        for result in unique_results[:top_k]:
            # 确保recipe_name字段正确设置
            recipe_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知菜品")
            
            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,  # 确保有recipe_name字段
                    "search_type": "dual_level",  # 设置搜索类型
                    **result.metadata
                }
            )
            documents.append(doc)
            
        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents
    
    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """
        增强的向量检索：结合图信息
        """
        try:
            # 使用Milvus进行向量检索
            vector_docs = self.milvus_module.similarity_search(query, k=top_k*2)
            
            # 用图信息增强结果并转换为Document对象
            enhanced_docs = []
            for result in vector_docs:
                # 从Milvus结果创建Document对象
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")
                
                if node_id:
                    # 从图中获取邻居信息
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        # 将邻居信息添加到内容中
                        neighbor_info = f"\n相关信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info
                
                # 确保recipe_name字段正确设置
                recipe_name = metadata.get("recipe_name", "未知菜品")
                
                # 调试：打印向量得分
                vector_score = result.get("score", 0.0)
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")
                
                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": recipe_name,  # 确保有recipe_name字段
                        "score": vector_score,
                        "search_type": "vector_enhanced"
                    }
                )
                # 尝试扩展为 Parent 块（Parent-Child 策略）
                doc = self._replace_with_parent(doc)
                enhanced_docs.append(doc)
                
            return enhanced_docs[:top_k]
            
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []
    
    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取节点的邻居信息"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        混合检索：三路并行执行（双层检索 + 向量检索 + BM25检索）
        """
        import concurrent.futures

        logger.info(f"开始三路并行混合检索: {query}")

        # 三路并行执行不同检索策略
        dual_docs = []
        vector_docs = []
        bm25_docs = []

        def dual_search():
            nonlocal dual_docs
            try:
                dual_docs = self.dual_level_retrieval(query, top_k)
                logger.info(f"双层检索完成: {len(dual_docs)} 个结果")
            except Exception as e:
                logger.error(f"双层检索失败: {e}")
                dual_docs = []

        def vector_search():
            nonlocal vector_docs
            try:
                vector_docs = self.vector_search_enhanced(query, top_k)
                logger.info(f"向量检索完成: {len(vector_docs)} 个结果")
            except Exception as e:
                logger.error(f"向量检索失败: {e}")
                vector_docs = []

        def bm25_search_task():
            nonlocal bm25_docs
            try:
                bm25_docs = self.bm25_search(query, top_k)
                logger.info(f"BM25检索完成: {len(bm25_docs)} 个结果")
            except Exception as e:
                logger.error(f"BM25检索失败: {e}")
                bm25_docs = []

        # 使用线程池三路并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_dual = executor.submit(dual_search)
            future_vector = executor.submit(vector_search)
            future_bm25 = executor.submit(bm25_search_task)

            # 等待检索完成
            concurrent.futures.wait(
                [future_dual, future_vector, future_bm25], timeout=30
            )

        # Round-robin轮询合并（三路）
        merged_docs = []
        seen_doc_ids = set()
        max_len = max(len(dual_docs), len(vector_docs), len(bm25_docs))
        origin_len = len(dual_docs) + len(vector_docs) + len(bm25_docs)
        
        for i in range(max_len):
            # 第1路：双层检索结果
            if i < len(dual_docs):
                doc = dual_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "dual_level"
                    doc.metadata["round_robin_order"] = len(merged_docs)
                    doc.metadata["final_score"] = doc.metadata.get("relevance_score", 0.0)
                    merged_docs.append(doc)
            
            # 第2路：向量检索结果
            if i < len(vector_docs):
                doc = vector_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "vector_enhanced"
                    doc.metadata["round_robin_order"] = len(merged_docs)
                    vector_score = doc.metadata.get("score", 0.0)
                    similarity_score = max(0.0, 1.0 - vector_score) if vector_score <= 1.0 else 0.0
                    doc.metadata["final_score"] = similarity_score
                    merged_docs.append(doc)

            # 第3路：BM25检索结果
            if i < len(bm25_docs):
                doc = bm25_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "bm25"
                    doc.metadata["round_robin_order"] = len(merged_docs)
                    doc.metadata["final_score"] = 0.5  # BM25 无统一分数，给中间值
                    merged_docs.append(doc)
        
        # 取前top_k个结果
        final_docs = merged_docs[:top_k]
        
        logger.info(f"Round-robin合并：从总共{origin_len}个结果合并为{len(final_docs)}个文档")
        logger.info(f"三路混合检索完成，返回 {len(final_docs)} 个文档")
        return final_docs
        
    def check_entity_hit(self, query: str) -> Tuple[bool, List[str]]:
        """
        检查query中是否包含Neo4j知识库中已知的菜名或食材名。
        用于判断是否存在词汇gap——如果用户说了知识库里存在的实体名，
        说明不需要做HyDE扩写，直接检索即可。
        
        Args:
            query: 用户查询（可能已经过指代消解）
            
        Returns:
            (has_hit, matched_entities):
            - has_hit: True表示query中包含已知实体
            - matched_entities: 命中的实体名称列表
        """
        if not self.driver:
            logger.warning("Neo4j未连接，跳过实体命中检测")
            return False, []
        
        try:
            # 清洗query：去除Lucene全文索引的特殊字符，避免parser报错
            import re as _re
            clean_query = _re.sub(r'[\+\-\!\&\|\(\)\{\}\[\]\^\"\~\*\?\:\\\/\n]', ' ', query)
            clean_query = _re.sub(r'\s+', ' ', clean_query).strip()
            if not clean_query:
                return False, []
            
            with self.driver.session() as session:
                # 菜名精确匹配：query中必须包含完整菜名才算"实体命中"
                # 不用全文索引（CJK分词匹配太宽，如"大肠"会命中"腊肠"）
                recipe_result = session.run("""
                    MATCH (r:Recipe)
                    WHERE size(r.name) >= 2
                    AND $query CONTAINS r.name
                    RETURN r.name AS name
                    ORDER BY size(r.name) DESC
                    LIMIT 5
                """, {"query": query})
                
                recipe_hits = [record["name"] for record in recipe_result]
                
                # 食材名匹配（仅记录，不参与entity_hit判断）
                # 食材CONTAINS噪音太大（如"九转大肠"会命中"大肠"），
                # 不用于决定是否触发HyDE
                ingredient_result = session.run("""
                    MATCH (i:Ingredient)
                    WHERE $query CONTAINS i.name
                    AND size(i.name) >= 2
                    AND NOT i.name =~ '[\\u0000-\\u007f]+'
                    RETURN i.name AS name
                    LIMIT 5
                """, {"query": query})
                
                ingredient_hits = [record["name"] for record in ingredient_result]
                
                all_hits = recipe_hits + ingredient_hits
                # entity_hit仅由菜名全文索引决定（食材CONTAINS噪音太大，不参与）
                has_hit = len(recipe_hits) > 0
                
                if has_hit:
                    logger.info(f"实体命中检测: 命中 {len(all_hits)} 个实体: {all_hits[:5]}")
                else:
                    logger.info(f"实体命中检测: 未命中，query与知识库存在词汇gap")
                    
                return has_hit, all_hits
                
        except Exception as e:
            logger.warning(f"实体命中检测失败: {e}，假设未命中，允许触发HyDE")
            return False, []  # 失败时返回未命中，允许HyDE兜底
    
    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭") 