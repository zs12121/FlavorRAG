"""
Web服务处理模块
负责处理Web API和静态文件服务
"""

import logging
import json
import time
import re
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, Optional

from rag_modules.observability import create_trace

logger = logging.getLogger(__name__)

class WebServiceHandler:
    """
    Web服务处理器
    
    功能：
    1. API路由处理
    2. 静态文件服务
    3. 错误处理
    4. 响应格式化
    """
    
    def __init__(self, rag_system):
        """初始化Web服务处理器"""
        self.rag_system = rag_system
        self.app = None
    
    def setup_flask_app(self):
        """设置Flask应用和路由"""
        try:
            from flask import Flask, request, jsonify, Response
            from flask_cors import CORS
            
            self.app = Flask(__name__)
            CORS(self.app)
            
            # 设置路由
            self._setup_routes()
            
            return self.app
            
        except ImportError as e:
            logger.error(f"Flask导入失败: {e}")
            return None
    
    def _setup_routes(self):
        """设置所有API路由"""
        from flask import request, jsonify, Response, send_from_directory
        
        @self.app.route('/')
        def serve_index():
            """提供主页"""
            return self._serve_static_file('index.html')
        
        @self.app.route('/<path:filename>')
        def serve_static(filename):
            """提供静态文件服务"""
            return self._serve_static_file(filename)
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查端点"""
            return jsonify({
                "status": "healthy",
                "timestamp": str(datetime.now()),
                "service": "RAG System"
            })
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """聊天API - 普通响应"""
            return self._handle_chat_request()
        
        @self.app.route('/api/chat/stream', methods=['POST'])
        def chat_stream():
            """聊天API - 流式响应"""
            return self._handle_stream_request()
        
        @self.app.route('/api/recipes/recommendations', methods=['POST'])
        def get_recommendations():
            """获取菜谱推荐"""
            return self._handle_recommendations_request()
        
        @self.app.route('/api/recipes/<recipe_id>', methods=['GET'])
        def get_recipe_detail(recipe_id):
            """获取菜谱详情"""
            return self._handle_recipe_detail_request(recipe_id)
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """获取系统统计信息"""
            return self._handle_stats_request()
        
        @self.app.route('/api/feedback', methods=['POST'])
        def submit_feedback():
            """用户反馈接口"""
            return self._handle_feedback_request()
    
    def _serve_static_file(self, filename):
        """提供静态文件服务"""
        import os
        from flask import send_from_directory
        
        # 安全检查，防止路径遍历攻击
        if '..' in filename or filename.startswith('/'):
            return "Forbidden", 403
        
        # 前端文件路径
        frontend_path = os.path.join(os.getcwd(), 'frontend', 'dist')
        
        try:
            if filename == 'index.html' or filename == '':
                return send_from_directory(frontend_path, 'index.html')
            else:
                return send_from_directory(frontend_path, filename)
        except FileNotFoundError:
            # 如果文件不存在，返回index.html（用于SPA路由）
            return send_from_directory(frontend_path, 'index.html')
    
    def _handle_chat_request(self):
        """处理普通聊天请求"""
        from flask import request, jsonify
        
        try:
            data = request.get_json()
            query = data.get('message', '')
            session_id = data.get('session_id', '')
            
            if not query:
                return jsonify({"error": "消息不能为空"}), 400
            
            # ========== Langfuse 全链路追踪 ==========
            trace = create_trace(
                name="chat",
                session_id=session_id,
                metadata={"query": query, "api": "chat"},
            )

            # 🚀 缓存检查和预处理
            # 指代词检测：有指代词时跳过缓存（指代词query不应命中历史缓存）
            has_pronoun = self.rag_system.query_rewriter._detect_pronoun_fast(query)
            
            with trace.span("cache_check") as cache_span:
                cached_response = None
                enhanced_query = query
                
                def check_cache():
                    nonlocal cached_response
                    cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
                
                def prepare_query():
                    nonlocal enhanced_query
                    enhanced_query = self.rag_system.cache_manager.get_context_for_query(session_id, query)
                
                if has_pronoun:
                    # 有指代词 → 跳过缓存，只做上下文预处理
                    cache_span.set_metadata({"hit": False, "skipped": True, "reason": "pronoun_detected"})
                    prepare_query()
                else:
                    # 无指代词 → 正常并行缓存检查和预处理
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_cache = executor.submit(check_cache)
                        future_query = executor.submit(prepare_query)
                        
                        # 等待缓存检查完成
                        concurrent.futures.wait([future_cache], timeout=1)
                        
                        if cached_response:
                            # 缓存命中，取消查询预处理
                            future_query.cancel()
                            cache_span.set_metadata({"hit": True, "similarity": "N/A"})
                            self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                            # 缓存命中也更新实体栈（缓存命中时没有消解信息）
                            self._update_entity_stack_after_turn(session_id, enhanced_query, cached_response, was_resolved=False)
                            trace.flush()
                            return jsonify({
                                "response": cached_response,
                                "query": query,
                                "session_id": session_id,
                                "timestamp": str(datetime.now()),
                                "from_cache": True
                            })
                        
                        # 缓存未命中，等待查询预处理完成
                        concurrent.futures.wait([future_query], timeout=2)
                    
                    cache_span.set_metadata({"hit": False})
            
            # ========== 查询改写（指代消解 — 用原始query，不用含历史的 enhanced_query）==========
            with trace.span("query_rewrite") as rewrite_span:
                rewrite_result = self.rag_system.query_rewriter.rewrite_query(
                    query=query,
                    session_id=session_id,
                    cache_manager=self.rag_system.cache_manager
                )
                rewritten_query = rewrite_result["rewritten_query"]
                entity_hit = rewrite_result["entity_hit"]
                rewrite_span.set_metadata({
                    "was_resolved": rewrite_result["was_resolved"],
                    "entity_hit": entity_hit,
                    "matched_entities": rewrite_result.get("matched_entities", []),
                    "rewritten": rewritten_query if rewrite_result["was_resolved"] else query,
                })

            # ========== 构建生成用的 query：检索用干净query，生成用含历史上下文的 query ==========
            if rewrite_result["was_resolved"] and enhanced_query != query:
                # 把 enhanced_query 末尾 "当前问题: {原始query}" 替换为消解后的
                idx = enhanced_query.rfind("当前问题:")
                if idx >= 0:
                    gen_query = enhanced_query[:idx] + f"当前问题: {rewritten_query}"
                else:
                    gen_query = enhanced_query
            else:
                gen_query = enhanced_query if enhanced_query != query else rewritten_query

            # 缓存未命中，执行完整的RAG流程 —— 检索用干净query
            with trace.span("retrieval") as retrieval_span:
                documents, analysis, pre_rerank_docs = self.rag_system.query_router.route_query(
                    query=rewritten_query,
                    top_k=self.rag_system.config.top_k
                )
                retrieval_span.set_metadata({
                    "strategy": analysis.recommended_strategy.value,
                    "complexity": round(analysis.query_complexity, 2),
                    "entity_count": analysis.entity_count,
                    "doc_count": len(documents),
                })
                retrieval_span.set_output({
                    "pre_rerank_count": len(pre_rerank_docs),
                    "pre_rerank_docs": [
                        {"title": getattr(d, 'title', ''), "content": d.page_content[:300]}
                        for d in pre_rerank_docs[:10]
                    ],
                    "final_doc_count": len(documents),
                    "strategy": analysis.recommended_strategy.value,
                })
            # HyDE动态判断
            if self.rag_system.config.enable_query_rewriting and \
               self.rag_system.query_rewriter.should_trigger_hyde(
                   entity_hit=entity_hit, documents=documents
               ):
                with trace.span("hyde_expansion") as hyde_span:
                    documents = self._do_hyde_search(
                        rewritten_query, documents, self.rag_system.config.top_k
                    )
                    hyde_span.set_metadata({
                        "doc_count_after_hyde": len(documents),
                    })

            # 使用生成模块生成最终答案 —— 用含历史上下文的 gen_query
            with trace.span("answer_generation") as gen_span:
                response = self.rag_system.generation_module.generate_adaptive_answer(gen_query, documents)
                gen_span.set_metadata({
                    "model": self.rag_system.generation_module.model_name,
                    "doc_count": len(documents),
                    "response_length": len(response),
                })
            
            # 将结果添加到会话缓存和上下文（用原始query做缓存键）
            # 指代消解的query不应写入语义缓存（如"第一个怎么做"的答案对"第二个怎么做"无意义）
            with trace.span("post_process") as post_span:
                if not rewrite_result["was_resolved"]:
                    self.rag_system.cache_manager.add_to_semantic_cache(query, response, session_id)
                self.rag_system.cache_manager.add_to_context(session_id, query, response)
                # 更新实体栈
                self._update_entity_stack_after_turn(
                    session_id, rewritten_query, response,
                    was_resolved=rewrite_result["was_resolved"],
                    documents=documents
                )
                post_span.set_metadata({"was_resolved": rewrite_result["was_resolved"]})
            
            # ========== RAGAS 数据收集 ==========
            # 收集评估数据用于后续批量分析
            contexts = [doc.page_content for doc in documents] if documents else []
            self.rag_system.ragas_collector.collect(
                question=query,
                contexts=contexts,
                answer=response,
                trace_id=trace.trace_id
            )
            
            trace.flush()
            return jsonify({
                "response": response,
                "query": query,
                "timestamp": str(datetime.now()),
                "trace_id": trace.trace_id
            })
            
        except Exception as e:
            logger.error(f"Chat API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_stream_request(self):
        """处理流式聊天请求"""
        from flask import request, Response
        
        try:
            data = request.get_json()
            query = data.get('message', '')
            session_id = data.get('session_id', '')
            
            if not query:
                return jsonify({"error": "消息不能为空"}), 400
            
            def generate():
                trace = None
                try:
                    # ========== Langfuse 全链路追踪 ==========
                    trace = create_trace(
                        name="chat_stream",
                        session_id=session_id,
                        metadata={"query": query, "api": "stream"},
                    )
                    trace.set_input({"query": query})

                    # 🚀 缓存检查和预处理
                    # 指代词检测：有指代词时跳过缓存（指代词query不应命中历史缓存）
                    has_pronoun = self.rag_system.query_rewriter._detect_pronoun_fast(query)
                    
                    with trace.span("cache_check") as cache_span:
                        cached_response = None
                        enhanced_query = query
                        
                        def check_cache():
                            nonlocal cached_response
                            cached_response = self.rag_system.cache_manager.check_semantic_cache(query, session_id)
                        
                        def prepare_query():
                            nonlocal enhanced_query
                            enhanced_query = self.rag_system.cache_manager.get_context_for_query(session_id, query)
                        
                        if has_pronoun:
                            # 有指代词 → 跳过缓存，只做上下文预处理
                            cache_span.set_metadata({"hit": False, "skipped": True, "reason": "pronoun_detected"})
                            prepare_query()
                        else:
                            # 无指代词 → 正常并行缓存检查和预处理
                            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                                future_cache = executor.submit(check_cache)
                                future_query = executor.submit(prepare_query)
                                
                                # 等待缓存检查完成
                                concurrent.futures.wait([future_cache], timeout=1)
                                
                                if cached_response:
                                    # 缓存命中，快速返回
                                    future_query.cancel()
                                    cache_span.set_metadata({"hit": True})
                                    self.rag_system.cache_manager.add_to_context(session_id, query, cached_response)
                                    # 缓存命中也更新实体栈（缓存命中时没有消解信息）
                                    self._update_entity_stack_after_turn(session_id, enhanced_query, cached_response, was_resolved=False)
                                    chunk_size = 3
                                    for i in range(0, len(cached_response), chunk_size):
                                        chunk = cached_response[i:i+chunk_size]
                                        data_obj = {"chunk": chunk, "from_cache": True}
                                        yield f"data: {json.dumps(data_obj)}\n\n"
                                        time.sleep(0.02)  # 更快的流式响应
                                    yield f"data: [DONE]\n\n"
                                    return
                                
                                # 缓存未命中，等待查询预处理完成
                                concurrent.futures.wait([future_query], timeout=2)
                            
                            cache_span.set_metadata({"hit": False})
                    
                    # ========== 查询改写（指代消解 — 用原始query，不用含历史的 enhanced_query）==========
                    with trace.span("query_rewrite") as rewrite_span:
                        rewrite_span.set_input({"original_query": query, "session_id": session_id})
                        rewrite_result = self.rag_system.query_rewriter.rewrite_query(
                            query=query,
                            session_id=session_id,
                            cache_manager=self.rag_system.cache_manager
                        )
                        rewritten_query = rewrite_result["rewritten_query"]
                        entity_hit = rewrite_result["entity_hit"]
                        rewrite_span.set_metadata({
                            "was_resolved": rewrite_result["was_resolved"],
                            "entity_hit": entity_hit,
                            "matched_entities": rewrite_result.get("matched_entities", []),
                            "rewritten": rewritten_query if rewrite_result["was_resolved"] else query,
                        })
                        rewrite_span.set_output({"rewritten_query": rewritten_query, "entity_hit": entity_hit})

                    # ========== 构建生成用的 query：检索用干净query，生成用含历史上下文的 query ==========
                    if rewrite_result["was_resolved"] and enhanced_query != query:
                        idx = enhanced_query.rfind("当前问题:")
                        if idx >= 0:
                            gen_query = enhanced_query[:idx] + f"当前问题: {rewritten_query}"
                        else:
                            gen_query = enhanced_query
                    else:
                        gen_query = enhanced_query if enhanced_query != query else rewritten_query

                    # 缓存未命中，执行完整的RAG流程 —— 检索用干净query
                    with trace.span("retrieval") as retrieval_span:
                        retrieval_span.set_input({"query": rewritten_query, "top_k": self.rag_system.config.top_k})
                        documents, analysis, pre_rerank_docs = self.rag_system.query_router.route_query(
                            query=rewritten_query,
                            top_k=self.rag_system.config.top_k
                        )
                        retrieval_span.set_metadata({
                            "strategy": analysis.recommended_strategy.value,
                            "complexity": round(analysis.query_complexity, 2),
                            "entity_count": analysis.entity_count,
                            "doc_count": len(documents),
                        })
                        retrieval_span.set_output({
                            "pre_rerank_count": len(pre_rerank_docs),
                            "pre_rerank_docs": [
                                {"title": getattr(d, 'title', ''), "content": d.page_content[:300]}
                                for d in pre_rerank_docs[:10]
                            ],
                            "final_doc_count": len(documents),
                            "strategy": analysis.recommended_strategy.value,
                        })

                    # HyDE动态判断
                    if self.rag_system.config.enable_query_rewriting and \
                       self.rag_system.query_rewriter.should_trigger_hyde(
                           entity_hit=entity_hit, documents=documents
                       ):
                        with trace.span("hyde_expansion") as hyde_span:
                            documents = self._do_hyde_search(
                                rewritten_query, documents, self.rag_system.config.top_k
                            )
                            hyde_span.set_metadata({
                                "doc_count_after_hyde": len(documents),
                            })
                    
                    # 流式生成答案 —— 用含历史上下文的 gen_query
                    with trace.span("answer_generation") as gen_span:
                        gen_span.set_input({"query": gen_query, "doc_count": len(documents)})
                        full_response = ""
                        for chunk in self.rag_system.generation_module.generate_adaptive_answer_stream(gen_query, documents):
                            full_response += chunk
                            data_obj = {"chunk": chunk}
                            yield f"data: {json.dumps(data_obj)}\n\n"
                        gen_span.set_metadata({
                            "model": self.rag_system.generation_module.model_name,
                            "doc_count": len(documents),
                            "response_length": len(full_response),
                        })
                        gen_span.set_output({"response_length": len(full_response)})
                    
                    # 将完整结果添加到会话缓存和上下文（用原始query做缓存键）
                    # 指代消解的query不应写入语义缓存
                    with trace.span("post_process") as post_span:
                        if not rewrite_result["was_resolved"]:
                            self.rag_system.cache_manager.add_to_semantic_cache(query, full_response, session_id)
                        self.rag_system.cache_manager.add_to_context(session_id, query, full_response)
                        # 更新实体栈
                        self._update_entity_stack_after_turn(
                            session_id, rewritten_query, full_response,
                            was_resolved=rewrite_result["was_resolved"],
                            documents=documents
                        )
                        post_span.set_metadata({"was_resolved": rewrite_result["was_resolved"]})
                    
                    # ========== RAGAS 数据收集 ==========
                    # 收集评估数据用于后续批量分析
                    contexts = [doc.page_content for doc in documents] if documents else []
                    self.rag_system.ragas_collector.collect(
                        question=query,
                        contexts=contexts,
                        answer=full_response,
                        trace_id=trace.trace_id
                    )
                    
                    # 发送结束标记（附带 trace_id 供前端反馈关联）
                    trace.set_output({"response_length": len(full_response)})
                    if trace.trace_id:
                        yield f"data: {json.dumps({'trace_id': trace.trace_id})}\n\n"
                    yield f"data: [DONE]\n\n"
                
                except Exception as e:
                    logger.error(f"Stream API错误: {e}")
                    error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                    data_obj = {"chunk": error_msg}
                    yield f"data: {json.dumps(data_obj)}\n\n"
                    yield f"data: [DONE]\n\n"
                finally:
                    if trace:
                        trace.flush()
            
            response = Response(generate(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
            
        except Exception as e:
            logger.error(f"Stream API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_recommendations_request(self):
        """处理菜谱推荐请求"""
        from flask import request, jsonify
        
        try:
            data = request.get_json() or {}
            preferences = data.get('preferences', {})
            
            # 获取推荐菜谱
            recipes = self.rag_system.recipe_manager.get_random_recipes_with_images(limit=3)
            
            return jsonify({
                "success": True,
                "data": recipes,
                "message": "推荐获取成功"
            })
            
        except Exception as e:
            logger.error(f"推荐API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_recipe_detail_request(self, recipe_id):
        """处理菜谱详情请求"""
        from flask import jsonify
        
        try:
            recipe = self.rag_system.recipe_manager.get_recipe_by_id(recipe_id)
            if recipe:
                return jsonify({
                    "success": True,
                    "data": recipe
                })
            else:
                return jsonify({"error": "菜谱不存在"}), 404
                
        except Exception as e:
            logger.error(f"菜谱详情API错误: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _handle_stats_request(self):
        """处理统计信息请求"""
        from flask import jsonify
        
        try:
            # 获取系统统计信息
            stats = {
                "cache_stats": self.rag_system.cache_manager.get_session_stats(),
                "route_stats": self.rag_system.query_router.get_route_statistics(),
                "system_info": {
                    "timestamp": str(datetime.now()),
                    "status": "running"
                }
            }
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"统计API错误: {e}")
            return jsonify({"error": str(e)}), 500

    def _handle_feedback_request(self):
        """处理用户反馈请求"""
        from flask import request, jsonify
        import os

        try:
            data = request.get_json()
            message_id = data.get('message_id', '')
            feedback_type = data.get('type')  # 'like' / 'dislike' / None（取消）
            session_id = data.get('session_id', '')
            trace_id = data.get('trace_id')

            if not message_id or feedback_type is None:
                return jsonify({"error": "参数不完整"}), 400

            # 1. 写入 JSONL（追加模式，文件不存在会自动创建）
            feedback_dir = os.path.join(os.getcwd(), 'data')
            os.makedirs(feedback_dir, exist_ok=True)
            feedback_file = os.path.join(feedback_dir, 'feedback.jsonl')

            feedback_record = {
                "message_id": message_id,
                "type": feedback_type,
                "session_id": session_id,
                "trace_id": trace_id,
                "timestamp": datetime.now().isoformat()
            }
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_record, ensure_ascii=False) + '\n')

            # 2. 尝试更新 Langfuse score（失败不影响主流程）
            if trace_id:
                try:
                    from rag_modules.observability import _get_client
                    client = _get_client()
                    if client:
                        value = 1.0 if feedback_type == 'like' else 0.0
                        client.create_score(
                            trace_id=trace_id,
                            name="user_feedback",
                            value=value
                        )
                        logger.info(f"Langfuse score 已更新: trace_id={trace_id}, value={value}")
                except Exception as e:
                    logger.warning(f"Langfuse score 更新失败: {e}")

            return jsonify({"success": True})

        except Exception as e:
            logger.error(f"反馈API错误: {e}")
            return jsonify({"error": str(e)}), 500

    # ============================================================
    # HyDE搜索 + 实体栈维护 辅助方法
    # ============================================================
    def _do_hyde_search(self, rewritten_query: str, original_documents, top_k: int):
        """
        HyDE假设文档扩写检索。
        1. 生成假设文档
        2. Embed后向量检索
        3. 与原始结果合并去重
        4. 漏斗精排
        """
        from langchain_core.documents import Document

        try:
            # 生成假设文档
            hyde_doc = self.rag_system.query_rewriter.generate_hyde_document(rewritten_query)
            if hyde_doc == rewritten_query:
                logger.warning("HyDE生成失败，回退到原始结果")
                return original_documents

            # Embed假设文档 → 向量检索
            hyde_embedding = self.rag_system.index_module.embeddings.embed_documents([hyde_doc])[0]
            hyde_results = self.rag_system.index_module.similarity_search(hyde_doc, k=top_k)

            # 转换为Document列表
            hyde_documents = []
            for result in hyde_results:
                doc = Document(
                    page_content=result.get("text", ""),
                    metadata={
                        "node_id": result.get("id", ""),
                        "relevance_score": result.get("score", 0.0),
                        "search_type": "hyde_enhanced",
                        **{k: v for k, v in result.items()
                           if k not in ("id", "text", "score")}
                    }
                )
                hyde_documents.append(doc)

            # 合并去重
            seen_ids = set()
            for doc in original_documents:
                seen_ids.add(doc.metadata.get("node_id",
                                               hash(doc.page_content)))

            merged = list(original_documents)
            for doc in hyde_documents:
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(doc)

            if len(merged) <= len(original_documents):
                logger.info("HyDE未带回新文档，保持原始结果")
                return original_documents

            # 漏斗精排
            query_text = rewritten_query
            ranked_docs = self.rag_system.reranker.rerank(
                query=query_text,
                documents=merged,  # 传Document对象列表，非字符串
                top_k=top_k,
                keep_ratio=self.rag_system.config.cosine_filter_keep_ratio,
            )

            # 重建Document列表，保持重排序后的顺序
            ranked_set = {d.page_content for d in ranked_docs}
            final_docs = []
            for doc in merged:
                if doc.page_content in ranked_set:
                    final_docs.append(doc)
                    if len(final_docs) >= top_k:
                        break

            logger.info(
                f"HyDE增强完成: 原始{len(original_documents)}条 + "
                f"HyDE{len(hyde_documents)}条 → 合并{len(merged)}条 "
                f"→ 精排后{len(final_docs)}条"
            )
            return final_docs

        except Exception as e:
            logger.error(f"HyDE搜索失败: {e}，回退到原始结果")
            return original_documents

    def _update_entity_stack_after_turn(
        self, session_id: str, rewritten_query: str, response: str,
        was_resolved: bool = False, documents: "list | None" = None
    ):
        """一轮对话结束后，提取实体并更新实体栈
        
        从AI回答中提取推荐菜名。用多层锚点确保只匹配推荐列表中的菜名，
        兼容各种checkmark变体（✅ ✓ ✔ ☑等），不依赖精确Unicode匹配。
        """
        try:
            if not self.rag_system.config.enable_anaphora_resolution:
                return
            entities = []
            
            if response:
                # 垃圾词（含"选项"避免把"选项一"当菜名）
                _junk = {
                    '推荐', '建议', '选项', '说明', '注意', '提示', '示例', '综合',
                    '食材', '主料', '辅料', '调料', '步骤', '特点', '类别', '优势',
                    '可能性', '区别', '对比', '攻略', '技巧', '方案', '结论',
                    '主食', '凉菜', '热菜', '荤菜', '素菜', '小吃', '饮品',
                }
                # 功能词 — 不出现在菜名中的动词/虚词
                _bad_in_name = {
                    '需要', '可以', '建议', '添加', '使用', '包括', '准备', '制作',
                    '作为', '例如', '比如', '如果', '但是', '因为', '所以',
                    '没有', '不是', '已经', '根据', '选择', '搭配', '推荐',
                }
                seen = set()
                
                for line in response.split('\n'):
                    ls = line.strip()
                    if not ls:
                        continue
                    
                    # 锚点检测：行首必须是checkmark类符号 或 数字序号
                    # checkmark类：任何非ASCII非字母非数字的前导符号（✅ ✓ ✔ ☑ ❇ etc）
                    has_checkmark = (
                        len(ls) > 0 and
                        ord(ls[0]) > 127 and
                        not ls[0].isalpha() and
                        not ls[0].isdigit()
                    )
                    has_digit_prefix = bool(re.match(r'\d+[.、．]', ls))
                    
                    if not (has_checkmark or has_digit_prefix):
                        continue
                    
                    # 去掉前缀（checkmark 和/或 数字序号），保留菜名
                    # 例：✅ 1. → 空, ✅ → 空, 1. → 空
                    clean = re.sub(
                        r'^[^\w\s\x00-\x7F]*\s*(?:\d+[.、．]\s*)?',
                        '', ls
                    ).strip()
                    
                    # 先按分隔符切分，再用re.match匹配每段
                    # 避免re.search全文贪婪匹配到描述文字
                    segments = re.split(r'[：:，,；;。\s、]+', clean)
                    matched_name = None
                    for seg in segments:
                        m = re.match(r'([\u4e00-\u9fff]{2,12})[（(]', seg)
                        if m:
                            candidate = m.group(1)
                            if not any(w in candidate for w in _junk) and \
                               not any(w in candidate for w in _bad_in_name):
                                matched_name = candidate
                                break
                    
                    if matched_name:
                        name = matched_name
                    else:
                        # 兜底：无括号纯中文名格式
                        if '：' in clean or ':' in clean or '，' in clean or '。' in clean:
                            continue
                        if re.fullmatch(r'[\u4e00-\u9fff]{2,12}', clean) and \
                           not any(w in clean for w in _junk) and \
                           not any(w in clean for w in _bad_in_name):
                            name = clean
                        else:
                            continue
                    if name in seen:
                        continue
                    if any(w in name for w in _junk):
                        continue
                    seen.add(name)
                    entities.append(name)
                
                if entities:
                    logger.info(f"实体提取（从AI文本）: {entities}")
            
            # 兜底：从检索结果提取
            if not entities and documents:
                for doc in documents:
                    recipe_name = (
                        doc.metadata.get("recipe_name") or
                        doc.metadata.get("name")
                    )
                    if recipe_name and recipe_name not in entities:
                        s = str(recipe_name)
                        if re.fullmatch(r'[\u4e00-\u9fff]{2,12}', s):
                            entities.append(s)
                if entities:
                    logger.info(f"实体提取（从检索结果）: {entities}")
            
            # 消解成功但所有方式都提取不到实体时，从消解后query推导
            if was_resolved and not entities:
                deduped = re.sub(
                    r'(怎么做|做法|步骤|怎么烧|怎么炒|怎么炖|怎么煮|怎么蒸|怎样做|如何做|呢|吗|啊|吧)$',
                    '', rewritten_query
                ).strip()
                if deduped and len(deduped) >= 2 and len(deduped) < len(rewritten_query):
                    entities = [deduped]
                    logger.info(f"实体推导: 从消解query '{rewritten_query}' 推导实体: {entities}")
            
            self.rag_system.cache_manager.update_entity_stack(session_id, entities)
        except Exception as e:
            logger.warning(f"更新实体栈失败: {e}")
