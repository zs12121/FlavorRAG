"""
重排序模块 - 使用Cross-Encoder对初筛结果进行语义级精排

优化策略：三阶段"漏斗"精排
  阶段A — Embedding余弦粗筛（复用已加载的bge-small-zh-v1.5，0.3秒）
     过滤掉语义明显不相关的候选（余弦<0.3的直接砍掉）
  阶段B — Cross-Encoder深度精排（bge-reranker-base，仅对通过粗筛的候选打分）
     对(query, document)对逐一打分，深度理解语义关系
  阶段C — 输出top_k结果

简单查询（复杂度<0.3）直接跳过Cross-Encoder，仅用embedding余弦排序。
"""

import logging
from typing import List, Optional

import numpy as np

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RerankerModule:
    """
    检索结果重排序器（漏斗精排版）

    使用"Embedding粗筛 + Cross-Encoder精排"两阶段漏斗：
    - Embedding模型（bge-small-zh-v1.5）：免费粗筛，砍掉明显不相关的
    - Cross-Encoder模型（bge-reranker-base）：精细打分，只给真正有竞争力的候选
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        embedding_model=None
    ):
        """
        初始化重排序器（懒加载模式，首次调用时才加载Cross-Encoder模型）

        Args:
            model_name: Cross-Encoder模型名称
            embedding_model: 已加载的嵌入模型实例（如HuggingFaceEmbeddings），
                           用于余弦相似度粗筛。传入None则跳过粗筛，直接全量CE打分。
        """
        self.model_name = model_name
        self._model = None
        self._load_failed = False
        self.embedding_model = embedding_model
        if embedding_model is not None:
            logger.info(f"重排序器已创建（漏斗模式：embedding粗筛 + CE精排）: {model_name}")
        else:
            logger.info(f"重排序器已创建（纯CE模式，模型将在首次使用时加载）: {model_name}")

    # ============================================================
    # 阶段A：Embedding 余弦相似度粗筛
    # ============================================================
    def _cosine_prefilter(
        self,
        query: str,
        documents: List[Document],
        keep_ratio: float = 0.6,
        min_keep: int = 5
    ) -> List[Document]:
        """
        使用已加载的embedding模型计算query与每个doc的余弦相似度，
        保留相似度最高的前keep_ratio比例（至少保留min_keep个）。
        
        目的：砍掉语义明显不相关的候选，减少CE打分数量。
        代价：零（embedding模型已在内存中，矩阵运算极快）。
        """
        if self.embedding_model is None:
            return documents
        if len(documents) <= min_keep:
            return documents

        try:
            # 批量编码query和documents
            try:
                query_vec = np.array(self.embedding_model.embed_query(query))
            except AttributeError:
                # 兼容没有embed_query方法的旧版
                query_vec = np.array(self.embedding_model.embed_documents([query])[0])

            doc_texts = [doc.page_content[:500] for doc in documents]  # 截断长文本
            doc_vecs = np.array(self.embedding_model.embed_documents(doc_texts))

            # 计算余弦相似度
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(doc_norms, query_norm)

            # 计算保留数量
            keep_count = max(min_keep, int(len(documents) * keep_ratio))
            keep_count = min(keep_count, len(documents))

            # 按相似度排序，保留前keep_count个
            top_indices = np.argsort(similarities)[-keep_count:][::-1]

            filtered = [documents[i] for i in top_indices]

            # 记录被砍掉的候选的相似度（用于日志分析）
            removed_indices = sorted(set(range(len(documents))) - set(top_indices.tolist()))
            if removed_indices:
                removed_sims = [f"{similarities[i]:.2f}" for i in removed_indices[:3]]
                logger.info(
                    f"Embedding粗筛: {len(documents)}→{len(filtered)} "
                    f"(砍掉余弦值: {', '.join(removed_sims)})"
                )

            return filtered

        except Exception as e:
            logger.warning(f"Embedding粗筛失败: {e}，跳过粗筛直接全量CE打分")
            return documents

    # ============================================================
    # 阶段B：Cross-Encoder 深度精排
    # ============================================================
    def _ensure_model_loaded(self):
        """确保Cross-Encoder模型已加载（懒加载）"""
        if self._model is not None:
            return
        if self._load_failed:
            return

        logger.info(f"正在加载重排序模型: {self.model_name}")
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"重排序模型加载完成: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers库未安装，跳过CE精排，使用embedding排序")
            self._load_failed = True
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}，降级为embedding排序")
            self._load_failed = True

    # ============================================================
    # 对外接口：漏斗精排
    # ============================================================
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        keep_ratio: float = 0.6
    ) -> List[Document]:
        """
        漏斗精排：embedding粗筛 → Cross-Encoder精排 → 输出top_k

        Args:
            query: 用户查询
            documents: 初筛的候选文档列表
            top_k: 最终返回的文档数量
            keep_ratio: embedding粗筛保留比例

        Returns:
            重排序后的文档列表，每个文档的metadata中会添加rerank_score字段
        """
        if not documents:
            logger.warning("重排序输入为空，跳过")
            return documents

        if len(documents) <= top_k:
            logger.info(
                f"候选文档数量({len(documents)})不超过top_k({top_k})，跳过重排序"
            )
            return documents

        try:
            # ========== 阶段A：Embedding余弦粗筛 ==========
            documents = self._cosine_prefilter(
                query, documents, keep_ratio=keep_ratio
            )

            # 粗筛后若已不超过top_k，直接返回
            if len(documents) <= top_k:
                logger.info(
                    f"Embedding粗筛后仅剩{len(documents)}个候选，跳过CE精排"
                )
                return documents

            # ========== 阶段B：Cross-Encoder深度精排 ==========
            self._ensure_model_loaded()

            if self._model is None:
                logger.info("CE模型不可用，使用embedding排序结果")
                return documents[:top_k]

            logger.info(
                f"Cross-Encoder精排: {len(documents)}个候选 → {top_k}个结果"
            )

            # 构造(query, document)对
            pairs = [[query, doc.page_content] for doc in documents]

            # Cross-Encoder逐对打分
            scores = self._model.predict(pairs)

            # 将文档和分数配对
            scored_docs = list(zip(documents, scores))

            # 按分数降序排列
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 取前top_k个，将分数写入metadata
            reranked = []
            for doc, score in scored_docs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                reranked.append(doc)

            logger.info(
                f"重排序完成: 最高CE分={scores.max():.4f}, "
                f"入选最低CE分={scored_docs[top_k - 1][1]:.4f}"
            )

            return reranked

        except Exception as e:
            logger.error(f"重排序失败: {e}，降级返回原始结果的前{top_k}个")
            return documents[:top_k]

    # ============================================================
    # 轻量重排（简单查询专用，仅用embedding余弦排序）
    # ============================================================
    def simple_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        轻量级重排序：仅用embedding余弦相似度排序，不加载Cross-Encoder。

        适用于简单查询（复杂度<0.3），耗时<0.5秒。

        Args:
            query: 用户查询
            documents: 候选文档列表
            top_k: 返回数量

        Returns:
            按余弦相似度排序后的文档列表
        """
        if not documents:
            return documents
        if len(documents) <= top_k:
            return documents

        if self.embedding_model is None:
            # 没有embedding模型，保持原序
            return documents[:top_k]

        try:
            try:
                query_vec = np.array(self.embedding_model.embed_query(query))
            except AttributeError:
                query_vec = np.array(self.embedding_model.embed_documents([query])[0])

            doc_texts = [doc.page_content[:500] for doc in documents]
            doc_vecs = np.array(self.embedding_model.embed_documents(doc_texts))

            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(doc_norms, query_norm)

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            result = [documents[i] for i in top_indices]

            logger.info(
                f"轻量重排完成: {len(documents)}→{len(result)} "
                f"(最高余弦={similarities[top_indices[0]]:.4f})"
            )
            return result

        except Exception as e:
            logger.warning(f"轻量重排失败: {e}，返回原始顺序的前{top_k}个")
            return documents[:top_k]
