"""
基于图数据库的RAG系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Milvus配置
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "moonshot-v1-8k")

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 重排序配置
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    enable_rerank: bool = True
    rerank_candidate_multiplier: int = 2  # 初筛时多取top_k的倍数（从3降到2，配合漏斗精排）
    cosine_filter_keep_ratio: float = 0.6  # embedding粗筛保留比例（60%）
    simple_query_skip_rerank: bool = True  # 简单查询跳过Cross-Encoder，仅用embedding排序
    simple_query_complexity_threshold: float = 0.30  # 复杂度低于此值视为简单查询

    # 查询分析配置
    use_llm_query_analysis: bool = False  # 默认用规则分析（快速），True则用LLM分析（准确但慢）

    # 查询改写配置
    enable_query_rewriting: bool = True  # 查询改写总开关
    enable_anaphora_resolution: bool = True  # 指代消解（LLM优先，规则降级）
    coref_window_size: int = 3  # 指代消解滑动窗口大小（最近N轮对话）
    enable_hyde: bool = True  # HyDE假设文档扩写（仅在检索不足时触发）
    hyde_trigger_min_results: int = 3  # 检索结果少于此数触发HyDE
    hyde_trigger_min_score: float = 0.4  # 检索最高分低于此值触发HyDE
    hyde_document_max_tokens: int = 256  # HyDE假设文档最大token数

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度

    # Parent-Child 切分配置
    use_parent_child: bool = True  # 是否启用 Parent-Child 切分策略
    parent_chunk_size: int = 1000  # Parent 块最大字符数
    child_chunk_size: int = 250    # Child 块最大字符数
    child_chunk_overlap: int = 30  # Child 块重叠字符数

    # RAGAS 评估配置
    enable_ragas_eval: bool = True  # True=启用数据收集, False=禁用
    ragas_metrics: list = None  # 评估指标列表，None=使用默认指标
    ragas_batch_file: str = "data/ragas_batch.jsonl"  # 批量模式数据存储文件

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'reranker_model': self.reranker_model,
            'enable_rerank': self.enable_rerank,
            'rerank_candidate_multiplier': self.rerank_candidate_multiplier,
            'cosine_filter_keep_ratio': self.cosine_filter_keep_ratio,
            'simple_query_skip_rerank': self.simple_query_skip_rerank,
            'simple_query_complexity_threshold': self.simple_query_complexity_threshold,
            'use_llm_query_analysis': self.use_llm_query_analysis,
            'enable_query_rewriting': self.enable_query_rewriting,
            'enable_anaphora_resolution': self.enable_anaphora_resolution,
            'coref_window_size': self.coref_window_size,
            'enable_hyde': self.enable_hyde,
            'hyde_trigger_min_results': self.hyde_trigger_min_results,
            'hyde_trigger_min_score': self.hyde_trigger_min_score,
            'hyde_document_max_tokens': self.hyde_document_max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'use_parent_child': self.use_parent_child,
            'parent_chunk_size': self.parent_chunk_size,
            'child_chunk_size': self.child_chunk_size,
            'child_chunk_overlap': self.child_chunk_overlap,
            'enable_ragas_eval': self.enable_ragas_eval,
            'ragas_metrics': self.ragas_metrics if self.ragas_metrics else [
                'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'
            ],
            'ragas_batch_file': self.ragas_batch_file
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig() 