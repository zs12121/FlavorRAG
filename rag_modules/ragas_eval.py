"""
RAGAS 数据收集模块（项目环境运行）

职责：
- 收集每次问答的 {question, contexts, answer, trace_id}
- 写入 JSONL 文件，供 ragas-eval 环境批量评估

注意：
- 本模块不 import ragas（避免依赖冲突）
- ragas 评估在独立环境中运行
"""

import os
import json
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RagasDataCollector:
    """RAGAS 数据收集器（项目环境运行）"""
    
    def __init__(self, config):
        """
        Args:
            config: GraphRAGConfig 配置实例
        """
        self.config = config
        self.enabled = config.enable_ragas_eval
        self.batch_file = config.ragas_batch_file
        
        logger.info(f"RagasDataCollector 初始化: enabled={self.enabled}, batch_file={self.batch_file}")
    
    def collect(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        trace_id: Optional[str] = None,
    ):
        """
        收集评估数据并写入 JSONL。
        
        Args:
            question: 用户问题
            contexts: 检索到的文档内容列表
            answer: 生成的答案
            trace_id: Langfuse Trace ID（用于回写分数）
        """
        if not self.enabled:
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.batch_file), exist_ok=True)
            
            record = {
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "trace_id": trace_id,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(self.batch_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"RAGAS 数据已收集: question={question[:50]}..., trace_id={trace_id}")
            
        except Exception as e:
            logger.error(f"RAGAS 数据收集失败: {e}")
