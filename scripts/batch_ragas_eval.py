"""
RAGAS 批量评估脚本

使用方法：
    conda activate ragas-eval
    python scripts/batch_ragas_eval.py

功能：
1. 读取 data/ragas_batch.jsonl 中收集的评估数据
2. 使用 RAGAS 进行批量评估
3. 将评估分数写回 Langfuse（按 trace_id 关联）
4. 输出评估报告

注意：
- 此脚本必须在 ragas-eval 环境中运行（因为项目环境与 ragas 依赖冲突）
- 需要配置 Langfuse 环境变量才能回写分数
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 加载项目根目录的 .env 文件（override=True 强制覆盖系统环境变量）
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(project_root, '.env'), override=True)

# 强制关闭 httpx 的系统代理检测（Windows 代理会干扰阿里云 API 认证）
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["no_proxy"] = "*"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_batch_data(batch_file: str) -> List[Dict[str, Any]]:
    """读取批量评估数据"""
    if not os.path.exists(batch_file):
        logger.error(f"批量评估文件不存在: {batch_file}")
        return []
    
    records = []
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    
    logger.info(f"读取到 {len(records)} 条评估数据")
    return records


def evaluate_batch(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    使用 RAGAS 进行批量评估
    
    Returns:
        平均分数 {metric_name: avg_score}
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            context_precision,
        )
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        from datasets import Dataset
        
        # 配置评估用的大模型
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = os.getenv("LLM_MODEL", "qwen-plus")
        
        if not api_key:
            logger.error("OPENAI_API_KEY 未设置，请检查 .env 文件")
            return {}
        
        logger.info(f"api_key 存在: {api_key[:8]}..., 模型: {model}, base_url: {base_url}")
        
        # 创建 ChatOpenAI，显式传入 httpx client，绕过系统代理
        import httpx
        http_client = httpx.Client(proxy=None, timeout=120.0)
        async_http_client = httpx.AsyncClient(proxy=None, timeout=120.0)
        
        chat_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            http_client=http_client,
            http_async_client=async_http_client,
        )
        evaluator_llm = LangchainLLMWrapper(chat_model)
        
        # 构造 RAGAS 数据集
        data = {
            "question": [r["question"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "answer": [r["answer"] for r in records],
            "ground_truth": [r.get("ground_truth", "") for r in records],
        }
        dataset = Dataset.from_dict(data)
        
        # 只用 Faithfulness + Context Precision（无需嵌入模型，无兼容问题）
        metrics = [faithfulness, context_precision]
        
        # 执行评估
        logger.info("开始 RAGAS 评估...")
        result = evaluate(
            dataset, 
            metrics=metrics,
            llm=evaluator_llm,
        )
        
        # 提取平均分数（RAGAS 返回的每个指标是 list，需要计算均值）
        import numpy as np
        
        def _safe_mean(val):
            """安全计算均值，兼容 float 和 list"""
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, list):
                valid = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
                return float(np.mean(valid)) if valid else 0.0
            return 0.0
        
        avg_scores = {
            "ragas_faithfulness": _safe_mean(result["faithfulness"]),
            "ragas_context_precision": _safe_mean(result["context_precision"]),
        }
        avg_scores["ragas_overall"] = sum(avg_scores.values()) / len(avg_scores)
        
        return avg_scores
        
    except ImportError as e:
        logger.error(f"RAGAS 未安装，请在 ragas-eval 环境中运行: {e}")
        return {}
    except Exception as e:
        logger.error(f"RAGAS 评估失败: {e}")
        return {}


def write_scores_to_langfuse(records: List[Dict[str, Any]], scores: Dict[str, float]):
    """
    将分数写回 Langfuse（按 trace_id 关联）
    
    注意：这里将平均分数写入每个 trace，便于在 Langfuse 网页上查看
    """
    try:
        from langfuse import Langfuse
        
        # 初始化 Langfuse 客户端
        host = os.getenv("LANGFUSE_HOST")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        
        if not all([host, public_key, secret_key]):
            logger.warning("Langfuse 未配置，跳过分数回写")
            return
        
        client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        
        logger.info("开始将分数写回 Langfuse...")
        
        success_count = 0
        for record in records:
            trace_id = record.get("trace_id")
            if not trace_id:
                continue
            
            try:
                for metric_name, score in scores.items():
                    client.create_score(
                        trace_id=trace_id,
                        name=metric_name,
                        value=score
                    )
                success_count += 1
            except Exception as e:
                logger.warning(f"分数写入失败 (trace_id={trace_id}): {e}")
        
        logger.info(f"分数回写完成: {success_count}/{len(records)} 条成功")
        client.flush()
        
    except ImportError:
        logger.warning("langfuse 未安装，跳过分数回写")
    except Exception as e:
        logger.error(f"Langfuse 分数回写失败: {e}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("RAGAS 批量评估脚本")
    logger.info("=" * 60)
    
    # 配置文件路径
    batch_file = "data/ragas_batch.jsonl"
    
    # 读取数据
    records = load_batch_data(batch_file)
    if not records:
        logger.error("没有评估数据")
        return
    
    # 执行评估
    logger.info("开始批量评估...")
    avg_scores = evaluate_batch(records)
    
    if not avg_scores:
        logger.error("评估失败")
        return
    
    # 输出评估报告
    logger.info("=" * 60)
    logger.info("RAGAS 批量评估结果")
    logger.info("=" * 60)
    for metric_name, score in avg_scores.items():
        logger.info(f"{metric_name}: {score:.4f}")
    logger.info("=" * 60)
    
    # 写回 Langfuse
    write_scores_to_langfuse(records, avg_scores)
    
    # 评估成功后清空文件，避免下次重复计算
    with open(batch_file, 'w', encoding='utf-8') as f:
        pass
    logger.info(f"已清空 {batch_file}，避免重复评估")
    
    logger.info("批量评估完成！")


if __name__ == "__main__":
    main()
