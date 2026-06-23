"""
智能查询路由器
根据查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的信息查找
- 图RAG检索：适合复杂的关系推理和知识发现

查询分析策略：规则优先（精准关键词+模式匹配）→ LLM兜底（规则不明确时）
重排序策略：简单查询跳过Cross-Encoder → 漏斗精排（embedding粗筛+CE精排）
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略
    
@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    reasoning_required: bool  # 是否需要推理
    entity_count: int  # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float  # 推荐置信度
    reasoning: str  # 推荐理由

class IntelligentQueryRouter:
    """
    智能查询路由器
    
    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理（规则优先 + LLM兜底）
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 按需重排：简单查询跳过CE，复杂查询漏斗精排
    """
    
    def __init__(self, 
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,    # 图RAG检索模块
                 llm_client,
                 config,
                 reranker=None):         # 重排序模块（可选）
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        self.reranker = reranker
        
        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        分析查询特征，决定最佳检索策略。
        
        优先级：规则匹配（高置信度） > 关键词评分 > LLM分析（低置信度兜底）
        """
        logger.info(f"分析查询特征: {query}")
        
        # ========== 第一步：规则分析 ==========
        rule_result = self._rule_based_analysis(query)
        
        # 规则置信度高（≥0.70），直接返回，不调LLM API
        if rule_result.confidence >= 0.70:
            logger.info(
                f"规则分析命中 (置信度: {rule_result.confidence:.2f}): "
                f"策略={rule_result.recommended_strategy.value}"
            )
            return rule_result
        
        # 规则置信度低（<0.65），调用LLM兜底分析
        if self.config.use_llm_query_analysis:
            logger.info("规则分析置信度不足，调用LLM分析...")
            return self._llm_analyze_query(query)
        else:
            # 配置不允许LLM分析，返回规则结果（带低置信度警告）
            logger.warning(
                f"规则分析置信度偏低({rule_result.confidence:.2f})，"
                f"但use_llm_query_analysis=False，使用规则结果"
            )
            return rule_result
    
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """
        基于规则的查询分析（主方案）
        
        设计原则：
        1. 精确模式匹配优先：正则捕获常见查询句式
        2. 保守策略：存疑时归类为复杂查询，避免误判
        3. 置信度分级：模式匹配>0.80 / 关键词明确0.70-0.80 / 无法判断0.55-0.65
        """
        query_clean = query.strip()
        
        # ========================================================
        # 第一层：精确模式匹配（最高置信度 0.85-0.92）
        # ========================================================
        
        # ---- 复杂查询模式 ----
        complex_patterns = [
            # 因果推理（置信度 0.90）
            (r'为什么.*(会|要|用|不|是|能|可以)', 0.82, 0.78, SearchStrategy.GRAPH_RAG,
             "因果推理：需要解释原因"),
            (r'原因|导致|造成|为啥|为何', 0.72, 0.68, SearchStrategy.GRAPH_RAG,
             "因果分析：需要推理原因"),
            
            # 对比分析（置信度 0.92）
            (r'区别|不同|差异|对比|比较|哪个好|哪个更|有何不同|有什么不同', 0.85, 0.80, SearchStrategy.GRAPH_RAG,
             "对比分析：需要比较多个实体"),
            
            # 多实体关系查询（置信度 0.88）
            (r'.+和.+((的|有什么|是啥|是什么)(关系|区别|联系|搭配|共同点|异同))', 0.85, 0.78, SearchStrategy.GRAPH_RAG,
             "多实体关系：需要分析实体间关联"),
            (r'.+与.+((的|有什么|是啥|是什么)(关系|区别|联系|搭配|共同点|异同))', 0.85, 0.78, SearchStrategy.GRAPH_RAG,
             "多实体关系：需要分析实体间关联"),
            (r'.+和.+怎么(搭配|配|组合|配合)', 0.82, 0.72, SearchStrategy.GRAPH_RAG,
             "搭配推荐：需要关系推理"),
            
            # 条件/替代推理（置信度 0.80）
            (r'(如果|要是|假如).*(怎么|能不能|可以|替代|代替|替换)', 0.78, 0.68, SearchStrategy.GRAPH_RAG,
             "条件推理：假设场景需要推理"),
            (r'(能不能|可以)用.+(替代|代替|替换|换)', 0.78, 0.68, SearchStrategy.GRAPH_RAG,
             "替代推理：需要判断可替代性"),
            
            # 影响分析（置信度 0.80）
            (r'影响|作用|效果|后果|功效|好处|坏处|营养价值', 0.72, 0.65, SearchStrategy.GRAPH_RAG,
             "影响分析：需要评价效果"),
            
            # 综合特征（置信度 0.82）
            (r'(有什么|有哪些).*(特点|特色|特征|讲究|技巧|秘诀)', 0.75, 0.62, SearchStrategy.COMBINED,
             "综合特征：需要多维度信息"),
            
            # 多种条件的组合（置信度 0.82）
            (r'(适合|推荐).*(什么|哪些|什么菜|什么食材)', 0.72, 0.62, SearchStrategy.COMBINED,
             "条件推荐：需要筛选匹配"),
            (r'.*(适合|推荐).*(老人|小孩|孕妇|减肥|健身|素食|糖尿病)', 0.75, 0.62, SearchStrategy.COMBINED,
             "人群推荐：需要条件匹配"),
        ]
        
        # ---- 简单查询模式 ----
        simple_patterns = [
            # 直接菜谱查询（置信度 0.90）
            (r'^.{1,6}(怎么做|做法|步骤|怎么烧|怎么炒|怎么炖|怎么煮|怎么蒸)', 0.15, 0.10, SearchStrategy.HYBRID_TRADITIONAL,
             "直接菜谱查询：查找具体做法"),
            (r'^(怎么做|怎么烧|怎么炒|怎么炖|怎么煮|怎么蒸).{1,6}', 0.15, 0.10, SearchStrategy.HYBRID_TRADITIONAL,
             "直接菜谱查询：查找具体做法"),
            (r'^.{2,6}(的|家常)(做法|食谱|配方)', 0.15, 0.10, SearchStrategy.HYBRID_TRADITIONAL,
             "菜谱查找：查询配方"),
            
            # 列举类查询（置信度 0.88）
            (r'(有什么|有哪些|推荐几个|给我推荐).{0,4}(菜|做法|食谱|好吃的)', 0.18, 0.10, SearchStrategy.HYBRID_TRADITIONAL,
             "列举查询：简单的列表检索"),
            (r'(有什么|有哪些).*(素菜|荤菜|汤|凉菜|早餐|午餐|晚餐|甜品|小吃)', 0.20, 0.10, SearchStrategy.HYBRID_TRADITIONAL,
             "分类列举：按类别查询"),
            
            # 单个食材信息（置信度 0.88）
            (r'^.{1,4}(的营养|的热量|的功效|是什么菜|多少钱|怎么选)', 0.18, 0.08, SearchStrategy.HYBRID_TRADITIONAL,
             "食材信息：简单属性查询"),
        ]
        
        # 检查复杂模式（优先，防止被简单模式覆盖）
        for pattern, complexity, relation, strategy, reason in complex_patterns:
            if re.search(pattern, query_clean):
                entity_count = self._count_entities(query_clean)
                return QueryAnalysis(
                    query_complexity=complexity,
                    relationship_intensity=relation,
                    reasoning_required=True,
                    entity_count=entity_count,
                    recommended_strategy=strategy,
                    confidence=0.88,
                    reasoning=reason
                )
        
        # 检查简单模式
        for pattern, complexity, relation, strategy, reason in simple_patterns:
            if re.search(pattern, query_clean):
                return QueryAnalysis(
                    query_complexity=complexity,
                    relationship_intensity=relation,
                    reasoning_required=False,
                    entity_count=1,
                    recommended_strategy=strategy,
                    confidence=0.88,
                    reasoning=reason
                )
        
        # ========================================================
        # 第二层：关键词评分（未命中模式时使用）
        # ========================================================
        
        # 推理相关关键词（权重高）
        reasoning_keywords = [
            ("为什么", 3), ("原因", 3), ("导致", 3), ("造成", 3),
            ("区别", 3), ("不同", 3), ("对比", 3), ("比较", 3), ("差异", 3),
            ("哪个好", 3), ("哪个更", 3),
        ]
        
        # 关系相关关键词（权重中）
        relation_keywords = [
            ("关系", 2), ("联系", 2), ("相关", 2),
            ("搭配", 2), ("配", 2), ("组合", 2),
            ("影响", 2), ("作用", 2), ("效果", 2),
            ("特点", 1), ("特色", 1), ("特征", 1),
        ]
        
        # 过程相关关键词（权重低，可能复杂也可能简单）
        process_keywords = [
            ("如何", 1), ("怎么", 1), ("怎样", 1),
        ]
        
        # 计算加权分数
        total_score = 0
        for kw, weight in reasoning_keywords:
            if kw in query_clean:
                total_score += weight
        
        relation_score = 0
        for kw, weight in relation_keywords:
            if kw in query_clean:
                relation_score += weight
        
        process_score = 0
        for kw, weight in process_keywords:
            if kw in query_clean:
                process_score += weight
        
        combined_score = total_score + relation_score + process_score
        entity_count = self._count_entities(query_clean)
        
        # ========================================================
        # 第三层：根据分数判断策略
        # ========================================================
        
        # 高分 → 图RAG（明确需要推理）
        if total_score >= 3:
            return QueryAnalysis(
                query_complexity=0.78,
                relationship_intensity=0.72,
                reasoning_required=True,
                entity_count=entity_count,
                recommended_strategy=SearchStrategy.GRAPH_RAG,
                confidence=0.78,
                reasoning=f"关键词评分={total_score}，含明确推理/对比关键词，推荐图RAG"
            )
        
        # 中分 → 组合策略（有关系但不需要深度推理）
        if combined_score >= 2 or (relation_score >= 2 and entity_count >= 2):
            return QueryAnalysis(
                query_complexity=0.55,
                relationship_intensity=0.55,
                reasoning_required=total_score >= 1,
                entity_count=entity_count,
                recommended_strategy=SearchStrategy.COMBINED,
                confidence=0.72,
                reasoning=f"关键词评分={combined_score}，关系+多实体，推荐组合策略"
            )
        
        # 低分但有一定复杂度 → 传统混合检索（中等复杂度）
        if combined_score >= 1:
            return QueryAnalysis(
                query_complexity=0.40,
                relationship_intensity=0.30,
                reasoning_required=False,
                entity_count=entity_count,
                recommended_strategy=SearchStrategy.HYBRID_TRADITIONAL,
                confidence=0.72,
                reasoning=f"关键词评分={combined_score}，中等复杂度，推荐传统混合检索"
            )
        
        # 无关键词命中 → 简单查询（高置信度，因为确实看不出复杂特征）
        return QueryAnalysis(
            query_complexity=0.18,
            relationship_intensity=0.08,
            reasoning_required=False,
            entity_count=max(1, entity_count),
            recommended_strategy=SearchStrategy.HYBRID_TRADITIONAL,
            confidence=0.82,
            reasoning="未检测到复杂查询特征，判定为简单信息查找"
        )

    def _count_entities(self, query: str) -> int:
        """简单估算查询中的实体数量（基于字数密度）"""
        # 粗略估算：中文字数/3 ≈ 大概实体数量
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        return max(1, chinese_chars // 3)

    def _llm_analyze_query(self, query: str) -> QueryAnalysis:
        """
        LLM兜底分析（仅在规则无法明确判断时调用）
        """
        analysis_prompt = f"""
作为RAG系统的查询分析专家，请深度分析以下查询的特征：

查询：{query}

请从以下维度分析：

1. 查询复杂度 (0-1)：
   - 0.0-0.3: 简单信息查找（如：红烧肉怎么做？）
   - 0.4-0.7: 中等复杂度（如：川菜有哪些特色菜？）
   - 0.8-1.0: 高复杂度推理（如：为什么川菜用花椒而不是胡椒？）

2. 关系密集度 (0-1)：
   - 0.0-0.3: 单一实体信息（如：西红柿的营养价值）
   - 0.4-0.7: 实体间关系（如：鸡肉配什么蔬菜？）
   - 0.8-1.0: 复杂关系网络（如：川菜的形成与地理、历史的关系）

3. 推理需求：
   - 是否需要多跳推理？
   - 是否需要因果分析？
   - 是否需要对比分析？

4. 实体识别：
   - 查询中包含多少个明确实体？
   - 实体类型是什么？

基于分析推荐检索策略：
- hybrid_traditional: 适合简单直接的信息查找
- graph_rag: 适合复杂关系推理和知识发现
- combined: 需要两种策略结合

返回JSON格式：
{{
    "query_complexity": 0.6,
    "relationship_intensity": 0.8,
    "reasoning_required": true,
    "entity_count": 3,
    "recommended_strategy": "graph_rag",
    "confidence": 0.85,
    "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            
            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.7),
                reasoning=result.get("reasoning", "LLM分析")
            )
            
            logger.info(f"LLM分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM分析失败: {e}，使用规则降级")
            return self._rule_based_analysis(query)
    
    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis, List[Document]]:
        """
        智能路由查询到最适合的检索引擎
        
        返回: (最终文档, 查询分析, 重排前的候选文档)
        
        优化逻辑：
        - 简单查询（复杂度<阈值）：candidate_k=top_k，跳过Cross-Encoder
        - 复杂查询（复杂度≥阈值）：candidate_k=top_k×倍数，漏斗精排
        """
        logger.info(f"开始智能路由: {query}")
        
        # 1. 分析查询特征
        analysis = self.analyze_query(query)
        
        # 2. 更新统计
        self._update_route_stats(analysis.recommended_strategy)
        
        # 3. 判断查询类型 → 决定检索量级和重排策略
        use_rerank = self.reranker is not None and self.config.enable_rerank
        threshold = self.config.simple_query_complexity_threshold
        is_simple = analysis.query_complexity < threshold
        
        if use_rerank:
            skip_ce = is_simple and self.config.simple_query_skip_rerank
            if skip_ce:
                candidate_k = top_k
                logger.info(
                    f"简单查询 (复杂度={analysis.query_complexity:.2f}<{threshold})，"
                    f"候选数={candidate_k}，跳过Cross-Encoder重排序"
                )
            else:
                candidate_k = top_k * self.config.rerank_candidate_multiplier
                logger.info(
                    f"复杂查询 (复杂度={analysis.query_complexity:.2f}≥{threshold})，"
                    f"候选数={candidate_k}，启用漏斗精排"
                )
        else:
            candidate_k = top_k
        
        # 4. 根据策略执行检索（三种策略完全保留）
        documents = []
        
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, candidate_k)
                
            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, candidate_k)
                
            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(query, candidate_k)
            
            # 5. 重排序前：保留候选文档快照
            pre_rerank_docs = list(documents)
            
            if use_rerank and documents:
                skip_ce = (is_simple and self.config.simple_query_skip_rerank)
                
                if skip_ce:
                    # 简单查询：轻量embedding余弦排序（<0.5秒）
                    documents = self.reranker.simple_rerank(
                        query, documents, top_k
                    )
                else:
                    # 复杂查询：漏斗精排（embedding粗筛 + CE精排）
                    documents = self.reranker.rerank(
                        query, documents, top_k,
                        keep_ratio=self.config.cosine_filter_keep_ratio
                    )
            
            # 6. 结果后处理
            documents = self._post_process_results(documents, analysis)
            
            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis, pre_rerank_docs
            
        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            # 降级到传统检索
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis, documents
    
    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """
        组合搜索策略：并行执行传统检索和图RAG检索
        """
        import concurrent.futures
        import threading

        # 分配结果数量
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k

        # 🚀 并行执行两种检索
        traditional_docs = []
        graph_docs = []

        def traditional_search():
            nonlocal traditional_docs
            try:
                from rag_modules.observability import _get_client
                lf = _get_client()
                span = lf.start_span(name="combined.traditional") if lf else None
                try:
                    traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
                    logger.info(f"传统检索完成: {len(traditional_docs)} 个结果")
                    if span:
                        span.update(metadata={"source": "traditional", "doc_count": len(traditional_docs)})
                finally:
                    if span:
                        span.end()
            except Exception as e:
                logger.error(f"传统检索失败: {e}")
                traditional_docs = []

        def graph_search():
            nonlocal graph_docs
            try:
                from rag_modules.observability import _get_client
                lf = _get_client()
                span = lf.start_span(name="combined.graph_rag") if lf else None
                try:
                    graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)
                    logger.info(f"图RAG检索完成: {len(graph_docs)} 个结果")
                    if span:
                        span.update(metadata={"source": "graph_rag", "doc_count": len(graph_docs)})
                finally:
                    if span:
                        span.end()
            except Exception as e:
                logger.error(f"图RAG检索失败: {e}")
                graph_docs = []

        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_traditional = executor.submit(traditional_search)
            future_graph = executor.submit(graph_search)

            # 等待两个检索完成
            concurrent.futures.wait([future_traditional, future_graph], timeout=30)

        # 合并和去重
        combined_docs = []
        seen_contents = set()
        
        # 交替添加结果（Round-robin）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图RAG结果（通常质量更高）
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)
            
            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)
        
        return combined_docs[:top_k]
    
    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """
        结果后处理：根据查询分析优化结果
        """
        for doc in documents:
            # 添加路由信息到元数据
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })
        
        return documents
    
    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计"""
        self.route_stats["total_queries"] += 1
        
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }
    
    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)
        
        explanation = f"""
        查询路由分析报告
        
        查询：{query}
        
        特征分析：
        - 复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体关系' if analysis.relationship_intensity < 0.8 else '复杂关系网络'})
        - 推理需求：{'是' if analysis.reasoning_required else '否'}
        - 实体数量：{analysis.entity_count}
        
        推荐策略：{analysis.recommended_strategy.value}
        置信度：{analysis.confidence:.2f}
        
        决策理由：{analysis.reasoning}
        """
        
        return explanation
