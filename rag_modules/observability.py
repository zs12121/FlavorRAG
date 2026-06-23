"""
可观测性模块 — Langfuse 全链路追踪封装

通过环境变量控制启用/禁用：
  LANGFUSE_HOST        — Langfuse 服务地址 (例: https://jp.cloud.langfuse.com)
  LANGFUSE_PUBLIC_KEY  — 项目公钥 (pk-lf-...)
  LANGFUSE_SECRET_KEY  — 项目密钥 (sk-lf-...)

未配置时自动降级为 no-op，不影响业务逻辑。
"""

import os
import uuid
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_client: Optional[Any] = None
_client_attempted = False


def _get_client():
    """获取 Langfuse 客户端（懒加载单例，全局复用）"""
    global _client, _client_attempted
    
    # 如果已经成功初始化过，直接返回
    if _client is not None:
        return _client
    
    # 如果已经尝试过但失败了，不再重试（避免重复日志）
    if _client_attempted:
        return None
    
    _client_attempted = True

    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not all([host, public_key, secret_key]):
        logger.info("Langfuse 未配置（缺少 LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY），全链路追踪已禁用")
        return None

    try:
        import httpx
        # 自定义 httpx 客户端：跨国网络需要足够长的超时
        custom_http = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,    # 连接超时 30 秒
                write=30.0,      # 写入超时 30 秒
                read=60.0,       # 读取超时 60 秒
                pool=30.0,       # 连接池超时 30 秒
            )
        )
        from langfuse import Langfuse
        _client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            flush_at=1,          # 每条数据立刻发送，不等待批量
            flush_interval=0.1,  # 最短等待间隔 0.1 秒
            httpx_client=custom_http,
        )
        logger.info(f"Langfuse 已连接: {host}")
        return _client
    except ImportError:
        logger.warning("langfuse 包未安装，全链路追踪已禁用。运行 pip install langfuse 后重试。")
        return None
    except Exception as e:
        logger.warning(f"Langfuse 连接失败: {e}，追踪已禁用")
        return None
# ---------------------------------------------------------------------------

class _NoopSpan:
    """Langfuse 不可用时的无操作占位符"""
    def set_metadata(self, data: Dict): pass
    def set_input(self, data: Any): pass
    def set_output(self, data: Any): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class _SpanAccessor:
    """Span 元数据访问器（在 context manager 内部使用）"""
    def __init__(self, span):
        self._span = span
    def set_metadata(self, data: Dict):
        try:
            self._span.update(metadata=data)
        except Exception:
            pass
    def set_input(self, data: Any):
        try:
            self._span.update(input=data)
        except Exception:
            pass
    def set_output(self, data: Any):
        try:
            self._span.update(output=data)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Trace — 一次用户查询的顶层上下文
# ---------------------------------------------------------------------------

class Trace:
    """
    一次用户查询对应的 Trace。

    用法:
        trace = Trace("chat", session_id="abc", metadata={...})
        with trace.span("cache_check") as s:
            s.set_metadata({"hit": True})
        with trace.span("query_rewrite") as s:
            s.set_metadata({"was_resolved": True})
        trace.flush()
    """

    def __init__(self, name: str, session_id: str = "", metadata: Dict = None):
        self._client = _get_client()
        self._cm = None  # context manager for root span
        self._span = None  # 根 span
        self.trace_id: Optional[str] = None  # Langfuse trace ID，供前端反馈关联

        logger.info(f"Trace.__init__ 开始: name={name}, client={'已配置' if self._client else 'None'}")

        if self._client:
            try:
                # Langfuse v3: 必须用 start_as_current_span 才能获取 trace_id
                self._cm = self._client.start_as_current_span(name=name)
                self._span = self._cm.__enter__()
                if self._span:
                    self._span.update(metadata=metadata or {})
                    self.trace_id = self._client.get_current_trace_id()
                    logger.info(f"Langfuse trace 创建成功: trace_id={self.trace_id}")
            except Exception as e:
                logger.warning(f"创建 Trace '{name}' 失败: {e}")
                if self._cm:
                    try: self._cm.__exit__(None, None, None)
                    except Exception: pass
                self._cm = None
                self._span = None
        else:
            logger.info("Langfuse 客户端未配置，跳过 trace 创建")

    @contextmanager
    def span(self, name: str, metadata: Dict = None):
        """
        在当前 Trace 下创建子 Span（上下文管理器）。
        """
        if self._span:
            try:
                s = self._client.start_span(name=name)
                if metadata:
                    s.update(metadata=metadata)
                yield _SpanAccessor(s)
                return
            except Exception as e:
                logger.warning(f"创建 Span '{name}' 失败: {e}")
        yield _NoopSpan()

    def set_metadata(self, data: Dict):
        """更新 Trace 级别元数据"""
        if self._span:
            try:
                self._span.update(metadata=data)
            except Exception:
                pass

    def set_input(self, data: Any):
        """设置 Trace 输入"""
        if self._span:
            try:
                self._span.update(input=data)
            except Exception:
                pass

    def set_output(self, data: Any):
        """设置 Trace 输出"""
        if self._span:
            try:
                self._span.update(output=data)
            except Exception:
                pass

    def score(self, name: str, value: float):
        """
        向当前 Trace 写入评分。
        Langfuse 支持同一 Trace 多个不同 name 的 score，互不冲突。
        例如: user_feedback, ragas_faithfulness, ragas_overall 等。
        """
        if self._client and self.trace_id:
            try:
                self._client.create_score(
                    trace_id=self.trace_id,
                    name=name,
                    value=value
                )
                logger.debug(f"Langfuse score 写入成功: {name}={value}")
            except Exception as e:
                logger.warning(f"Langfuse score 写入失败 ({name}={value}): {e}")

    def flush(self):
        """结束 Trace 并确保数据上报"""
        if self._cm and self._span:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass
            self._cm = None
            self._span = None
        if self._client:
            try:
                logger.info("Langfuse flush() 开始...")
                self._client.flush()
                logger.info("Langfuse flush() 完成")
            except Exception as e:
                logger.error(f"Langfuse flush() 失败: {e}")

    def __del__(self):
        self.flush()


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """检查全链路追踪是否启用"""
    return _get_client() is not None


def create_trace(name: str, session_id: str = "", metadata: Dict = None) -> Trace:
    """创建一次查询 Trace（始终返回 Trace 对象，不可用时内部为 no-op）"""
    return Trace(name=name, session_id=session_id, metadata=metadata)
