"""
查询改写模块 — 指代消解 + HyDE假设文档扩写

动态判断流程：
  关1 — 指代消解（LLM优先，规则降级）
       检测指代词 → LLM推理消解 → 失败则降级到规则
  关2 — 实体命中检测（查Neo4j，~20ms）
       query中是否有已知菜名/食材名 → 有则直接检索，无则标记需要HyDE检查
  关3 — HyDE兜底（LLM，仅检索不足时触发）
       检索结果少或分数低 + 无实体命中 → 生成假设文档 → embed → 检索 → 合并
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ============================================================
# 指代词列表（用于快速检测，触发LLM消解）
# ============================================================

# 简单指代词
DEMONSTRATIVE_PRONOUNS = [
    "这个", "那个", "它", "这种", "那种", "这", "那",
]

# 序数指代
ORDINAL_PRONOUNS = [
    "第一个", "第二个", "第三个", "第四个", "第五个",
    "第一道", "第二道", "第三道", "第四道", "第五道",
    "第一款", "第二款", "第三款",
]

# 前文指代
PRIOR_PRONOUNS = [
    "前面那个", "前面说的", "刚才说的", "刚才那个",
    "之前那个", "之前说的", "上一个", "上一道",
    "刚才推荐", "前面推荐", "之前推荐",
]

# 替代指代
ALTERNATIVE_PRONOUNS = [
    "另一个", "别的", "其他的", "还有吗", "还有别的",
    "换一个", "换一道", "还有别的吗", "还有什么",
    "除了这个", "除了那个", "不要这个",
]

# 继续指代
CONTINUE_PRONOUNS = [
    "继续", "接着", "然后呢", "还有呢", "接下来呢",
    "再说说", "再说", "再推荐", "再介绍",
]

# 最初指代
FIRST_PRONOUNS = [
    "最开始那个", "第一个说的", "第一轮说的",
    "最早说的", "最开始说的",
]

# 描述性指代（新增）
DESCRIPTIVE_PRONOUNS = [
    "那个菜", "那道菜", "那个汤", "那道汤",
    "那个主食", "那个小吃", "那个点心",
    "带汤的", "辣的", "清淡的", "甜的", "咸的",
    "刚才说的那个", "之前说的那个",
]

# 比较指代（新增）
COMPARISON_PRONOUNS = [
    "和它比", "跟它比", "与它比",
    "和那个比", "跟那个比", "与那个比",
    "比它", "比那个",
]

# 所有指代词的全量集合（用于快速检测）
ALL_PRONOUNS = (
    DEMONSTRATIVE_PRONOUNS + ORDINAL_PRONOUNS +
    PRIOR_PRONOUNS + ALTERNATIVE_PRONOUNS +
    CONTINUE_PRONOUNS + FIRST_PRONOUNS +
    DESCRIPTIVE_PRONOUNS + COMPARISON_PRONOUNS
)

# 菜名提取黑名单（保留，用于降级路径）
_RECIPE_NAME_BLACKLIST = {
    "推荐", "建议", "选项", "说明", "注意", "提示", "示例", "综合",
    "火锅", "烤肉", "烧烤", "汤品", "凉菜", "热菜", "主食", "荤菜",
    "素菜", "小吃", "饮品", "点心", "套餐", "组合",
    "沪菜", "粤菜", "川菜", "湘菜", "鲁菜", "闽菜", "苏菜", "浙菜", "徽菜",
    "食材", "主料", "辅料", "调料", "步骤",
}


class QueryRewriter:
    """
    查询改写器

    集成指代消解和HyDE假设文档扩写，
    通过动态判断决定是否改写以及改写策略。
    """

    def __init__(
        self,
        llm_client,
        config,
        hybrid_retrieval,
        embedding_model=None
    ):
        """
        Args:
            llm_client: LLM客户端（用于HyDE生成假设文档）
            config: GraphRAGConfig配置实例
            hybrid_retrieval: HybridRetrievalModule实例（用于Neo4j实体命中检测）
            embedding_model: 已加载的嵌入模型（用于HyDE检索）
        """
        self.llm_client = llm_client
        self.config = config
        self.hybrid_retrieval = hybrid_retrieval
        self.embedding_model = embedding_model

    # ============================================================
    # 主入口
    # ============================================================
    def rewrite_query(
        self,
        query: str,
        session_id: str,
        cache_manager
    ) -> Dict[str, Any]:
        """
        查询改写主入口（在检索前调用）。

        流程：
        1. 快速检测指代词 → 无则跳过消解
        2. 有指代词 → LLM消解（优先） → 失败则规则降级 → 失败则用原始query
        3. 实体命中检测
        4. 返回改写结果

        Returns:
            {
                "rewritten_query": str,       # 最终用于检索的query
                "was_resolved": bool,         # 是否做了指代消解
                "entity_hit": bool,           # Neo4j实体命中检测结果
                "matched_entities": List[str],# 命中的实体名
            }
        """
        rewritten = query
        was_resolved = False
        matched_entities = []

        # ========== 关1：指代消解 ==========
        if self.config.enable_anaphora_resolution:
            # 快速检测是否有指代词
            has_pronoun = self._detect_pronoun_fast(query)
            
            if has_pronoun:
                # 有指代词 → 尝试LLM消解
                llm_resolved = self._resolve_anaphora_llm(query, session_id, cache_manager)
                
                if llm_resolved and llm_resolved != query:
                    # LLM消解成功（返回了不同的query）
                    rewritten = llm_resolved
                    was_resolved = True
                    logger.info(f"指代消解(LLM): '{query}' → '{rewritten}'")
                # LLM消解失败 或 LLM返回原始query（表示无法消解） → 降级到规则
                if not was_resolved:
                    entity_stack = cache_manager.get_entity_stack(session_id)
                    if entity_stack:
                        rule_resolved = self._resolve_anaphora(query, entity_stack)
                        if rule_resolved and rule_resolved != query:
                            rewritten = rule_resolved
                            was_resolved = True
                            logger.info(f"指代消解(规则降级): '{query}' → '{rewritten}'")

        # ========== 关2：实体命中检测 ==========
        entity_hit, matched_entities = self.hybrid_retrieval.check_entity_hit(rewritten)

        return {
            "rewritten_query": rewritten,
            "was_resolved": was_resolved,
            "entity_hit": entity_hit,
            "matched_entities": matched_entities,
        }

    # ============================================================
    # 快速指代词检测（轻量级，只判断有无）
    # ============================================================
    def _detect_pronoun_fast(self, query: str) -> bool:
        """
        快速检测query中是否包含指代词。
        只返回True/False，不做具体映射。
        """
        query_clean = query.strip()
        
        # 长度检查：中文>=8字通常不是纯指代
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query_clean))
        if chinese_chars >= 8:
            return False
        
        # 遍历所有指代词
        for pronoun in ALL_PRONOUNS:
            if pronoun in query_clean:
                return True
        
        return False

    # ============================================================
    # LLM指代消解（核心方法）
    # ============================================================
    def _resolve_anaphora_llm(
        self,
        query: str,
        session_id: str,
        cache_manager
    ) -> Optional[str]:
        """
        使用LLM进行指代消解。
        
        流程：
        1. 构建上下文（滑动窗口）
        2. 构造prompt（含few-shot示例）
        3. 调用LLM
        4. 解析输出
        
        Returns:
            消解后的query，失败返回None
        """
        try:
            # 构建上下文
            context = self._build_coref_context(query, session_id, cache_manager)
            
            # 构造prompt
            prompt = self._build_coref_prompt(context, query)
            
            # 调用LLM（设置超时）
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度确保稳定
                max_tokens=200,
                timeout=5,  # 5秒超时
            )
            
            llm_output = response.choices[0].message.content.strip()
            
            # 解析输出
            resolved_query = self._parse_llm_coref_output(llm_output)
            
            if resolved_query:
                logger.info(f"LLM指代消解成功: '{query}' → '{resolved_query}'")
                return resolved_query
            else:
                logger.warning(f"LLM指代消解输出解析失败: {llm_output[:100]}")
                return None
                
        except Exception as e:
            logger.warning(f"LLM指代消解失败: {e}")
            return None

    # ============================================================
    # 构建LLM coref专用上下文
    # ============================================================
    def _build_coref_context(
        self,
        current_query: str,
        session_id: str,
        cache_manager
    ) -> str:
        """
        构建LLM指代消解专用的上下文格式。
        
        格式：
        对话历史：
        第1轮：
          用户：xxx
          AI：xxx（截取前150字）
        
        第2轮：
          用户：xxx
          AI：xxx
        
        当前问题：xxx
        """
        try:
            if not session_id or session_id not in cache_manager.session_contexts:
                return f"当前问题：{current_query}"
            
            context_list = cache_manager.session_contexts[session_id]
            if not context_list:
                return f"当前问题：{current_query}"
            
            # 取最近N轮（滑动窗口）
            window_size = getattr(self.config, 'coref_window_size', 3)
            recent_context = context_list[-window_size:] if len(context_list) > window_size else context_list
            
            # 构建格式化的上下文
            parts = ["对话历史："]
            
            for i, item in enumerate(recent_context, 1):
                parts.append(f"第{i}轮：")
                parts.append(f"  用户：{item['query']}")
                # AI回答截取前500字（需确保不丢失关键实体信息，如菜名列表）
                ai_response = item['response'][:500]
                if len(item['response']) > 500:
                    ai_response += "..."
                parts.append(f"  AI：{ai_response}")
                parts.append("")  # 空行分隔
            
            parts.append(f"当前问题：{current_query}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.warning(f"构建coref上下文失败: {e}")
            return f"当前问题：{current_query}"

    # ============================================================
    # 构造LLM coref prompt
    # ============================================================
    def _build_coref_prompt(self, context: str, current_query: str) -> str:
        """
        构造LLM指代消解的prompt。
        
        包含：
        1. 任务说明
        2. 5个few-shot示例（针对项目场景）
        3. CoT推理指令
        4. 严格输出格式约束
        """
        prompt = f"""你是一个指代消解专家。请根据对话历史，判断用户当前问题中的指代词（如"这个""那个""第一个""它"等）具体指代什么，并输出消解后的完整query。

## 任务说明
1. 仔细阅读对话历史，理解上下文
2. 识别当前问题中的指代词
3. 推理指代词的具体指代对象
4. 输出消解后的完整query（用具体实体替换指代词）

## 示例

### 示例1：简单序数指代
对话历史：
第1轮：
  用户：今天晚餐吃什么
  AI：推荐三道菜：1. 宫保鸡丁（川菜） 2. 鱼香肉丝（川菜） 3. 麻婆豆腐（川菜）

当前问题：第一个怎么做

推理：用户说"第一个"，根据上下文，AI推荐的第一道菜是"宫保鸡丁"，所以"第一个"指代"宫保鸡丁"。
最终query：宫保鸡丁怎么做

### 示例2：描述性指代
对话历史：
第1轮：
  用户：有什么辣的菜推荐
  AI：推荐：1. 麻婆豆腐（麻辣） 2. 水煮鱼（麻辣） 3. 辣子鸡（香辣）

当前问题：那个加了花椒的

推理：用户说"那个加了花椒的"，根据上下文，麻婆豆腐是经典的麻辣菜，花椒是其主要调料，所以指代"麻婆豆腐"。
最终query：麻婆豆腐

### 示例3：跨轮指代
对话历史：
第1轮：
  用户：今天晚餐吃什么
  AI：推荐：1. 宫保鸡丁 2. 鱼香肉丝

第2轮：
  用户：鱼香肉丝怎么做
  AI：鱼香肉丝的做法是...

当前问题：它和第一个比哪个更辣

推理：用户说"它"指代上一轮讨论的"鱼香肉丝"，"第一个"指代第一轮推荐的"宫保鸡丁"。
最终query：鱼香肉丝和宫保鸡丁哪个更辣

### 示例4：继续追问
对话历史：
第1轮：
  用户：有什么清淡的菜
  AI：推荐：1. 清蒸鲈鱼 2. 白灼虾

当前问题：还有吗

推理：用户说"还有吗"，表示想要更多推荐，上下文是"清淡的菜"。
最终query：还有什么清淡的菜推荐

### 示例5：替代指代
对话历史：
第1轮：
  用户：今天晚餐吃什么
  AI：推荐：1. 宫保鸡丁 2. 鱼香肉丝

当前问题：换一个

推理：用户说"换一个"，表示不想要当前推荐的，想要其他选择。
最终query：除了宫保鸡丁和鱼香肉丝，还有什么推荐

## 你的任务

{context}

请按以下格式输出：
推理：<简要说明指代词的指代对象>
最终query：<消解后的完整query>

注意：
1. 如果当前问题中没有指代词，或无法从上下文推断，直接输出原始问题
2. 最终query必须是完整的、可独立理解的句子
3. 不要输出多余内容，严格按照格式输出"""

        return prompt

    # ============================================================
    # 解析LLM coref输出
    # ============================================================
    def _parse_llm_coref_output(self, llm_output: str) -> Optional[str]:
        """
        解析LLM指代消解的输出。
        
        期望格式：
        推理：xxx
        最终query：xxx
        
        Returns:
            提取的最终query，解析失败返回None
        """
        try:
            # 提取"最终query："后面的内容
            match = re.search(r'最终query[：:]\s*(.+)', llm_output)
            if match:
                resolved = match.group(1).strip()
                # 基本校验：非空、长度合理
                if resolved and len(resolved) >= 2 and len(resolved) <= 100:
                    return resolved
            
            logger.warning(f"无法从LLM输出中提取最终query: {llm_output[:100]}")
            return None
            
        except Exception as e:
            logger.warning(f"解析LLM coref输出失败: {e}")
            return None

    # ============================================================
    # 规则指代消解（降级路径，保留原有逻辑）
    # ============================================================
    def _resolve_anaphora(
        self,
        query: str,
        entity_stack: List[List[str]]
    ) -> Optional[str]:
        """
        基于规则的指代消解（降级路径）。

        触发条件：
        - query中包含指代词
        - query较短（<8个中文字，太长的query不会纯粹是指代）
        - 实体栈非空

        不触发则返回None，触发则返回改写后的query。
        """
        query_clean = query.strip()

        # 条件1：检测指代词
        matched_pronoun = self._detect_pronoun(query_clean)
        if matched_pronoun is None:
            return None

        # 条件2：query长度检查
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query_clean))
        if chinese_chars >= 8:
            logger.debug(f"query过长({chinese_chars}字)，跳过指代消解")
            return None

        # 条件3：实体栈非空
        if not entity_stack:
            return None

        # 根据指代词类型解析
        return self._map_pronoun_to_entity(matched_pronoun, query_clean, entity_stack)

    def _detect_pronoun(self, query: str) -> Optional[str]:
        """检测query中的指代词类型，返回匹配到的指代词字符串"""
        # 序数词（优先级高，因为"第二个"含"一个"）
        for p in ORDINAL_PRONOUNS:
            if p in query:
                return p
        # 时间指代
        for p in FIRST_PRONOUNS:
            if p in query:
                return p
        # 前文指代
        for p in PRIOR_PRONOUNS:
            if p in query:
                return p
        # 替代指代
        for p in ALTERNATIVE_PRONOUNS:
            if p in query:
                return p
        # 继续指代
        for p in CONTINUE_PRONOUNS:
            if p in query:
                return p
        # 描述性指代
        for p in DESCRIPTIVE_PRONOUNS:
            if p in query:
                return p
        # 比较指代
        for p in COMPARISON_PRONOUNS:
            if p in query:
                return p
        # 简单指代
        for p in DEMONSTRATIVE_PRONOUNS:
            if p in query:
                return p
        return None

    def _map_pronoun_to_entity(
        self,
        pronoun: str,
        query: str,
        entity_stack: List[List[str]]
    ) -> Optional[str]:
        """
        将指代词映射到实体栈中的具体实体。

        映射规则：
        - "第一个"/"第二个"… → stack[-1][0], stack[-1][1]…
        - "前面那个"/"刚才说的" → 从stack[-2]向下找第一个非空轮次[0]
        - "另一个"/"别的" → stack[-1]中未讨论过的
        - "继续" → stack[-1][0]
        - "最开始那个" → stack[0]的第一个非空元素
        - "这个"/"那个"/"它" → stack[-1][0]
        """
        try:
            # 序数指代
            if pronoun in ORDINAL_PRONOUNS:
                return self._resolve_ordinal(pronoun, query, entity_stack)

            # 前文指代
            if pronoun in PRIOR_PRONOUNS:
                return self._resolve_prior(query, entity_stack)

            # 替代指代
            if pronoun in ALTERNATIVE_PRONOUNS:
                return self._resolve_alternative(query, entity_stack)

            # 继续指代
            if pronoun in CONTINUE_PRONOUNS:
                if entity_stack[-1]:
                    return self._replace_pronoun(query, pronoun, entity_stack[-1][0])

            # 最初指代
            if pronoun in FIRST_PRONOUNS:
                for entities in entity_stack:
                    if entities:
                        return self._replace_pronoun(query, pronoun, entities[0])

            # 默认：简单指代 → 栈顶第一个实体
            if pronoun in DEMONSTRATIVE_PRONOUNS:
                if entity_stack[-1]:
                    return self._replace_pronoun(query, pronoun, entity_stack[-1][0])

        except Exception as e:
            logger.warning(f"指代映射失败: {e}")

        return None

    def _resolve_ordinal(
        self, pronoun: str, query: str, entity_stack: List[List[str]]
    ) -> Optional[str]:
        """解析序数指代：第一个、第二个…
        
        从栈顶向下搜索，找到第一个包含足够实体的轮次。
        例如：栈=[['A','B','C'], ['A']]，"第二个" → 回退到第一轮 → 'B'
        """
        ordinal_map = {
            "第一个": 0, "第二个": 1, "第三个": 2, "第四个": 3, "第五个": 4,
            "第一道": 0, "第二道": 1, "第三道": 2, "第四道": 3, "第五道": 4,
            "第一款": 0, "第二款": 1, "第三款": 2,
        }
        idx = ordinal_map.get(pronoun, 0)

        # 从后往前搜索：找到第一个有足够实体数的轮次
        for turn_entities in reversed(entity_stack):
            if idx < len(turn_entities):
                entity = turn_entities[idx]
                return self._replace_pronoun(query, pronoun, entity)
        return None

    def _resolve_prior(
        self, query: str, entity_stack: List[List[str]]
    ) -> Optional[str]:
        """解析前文指代：前面那个、刚才说的…"""
        # 从栈[-2]向下找第一个非空轮次
        pronoun = self._detect_pronoun(query) or "前面"
        for i in range(len(entity_stack) - 2, -1, -1):
            if entity_stack[i]:
                return self._replace_pronoun(query, pronoun, entity_stack[i][0])
        return None

    def _resolve_alternative(
        self, query: str, entity_stack: List[List[str]]
    ) -> Optional[str]:
        """解析替代指代：另一个、别的…"""
        pronoun = self._detect_pronoun(query) or "另一个"
        latest = entity_stack[-1]
        if len(latest) >= 2:
            # 返回第二个（假设第一个已被讨论）
            return self._replace_pronoun(query, pronoun, latest[1])
        return None

    def _replace_pronoun(self, query: str, pronoun: str, entity: str) -> str:
        """将query中的指代词替换为实体名"""
        if pronoun in query:
            return query.replace(pronoun, entity, 1)
        # 如果指代词不在query中（如"继续"→应该追加而非替换）
        return f"{entity}{query}"

    # ============================================================
    # HyDE 假设文档扩写
    # ============================================================

    def should_trigger_hyde(
        self,
        entity_hit: bool,
        documents: List[Document],
    ) -> bool:
        """
        判断是否需要触发HyDE假设文档扩写。

        触发条件：
        1. 全局开关 enable_hyde=True
        2. 无实体命中 —— query中不包含知识库里已有的菜名
           （即用户问的东西知识库里没有，需要HyDE桥接词汇gap）

        注意：不再检查检索质量（分数/数量阈值），因为检索系统对任何
        query都会返回结果，分数阈值的区分度太低。
        """
        if not self.config.enable_hyde:
            logger.info("HyDE: 全局开关关闭，跳过")
            return False
        if entity_hit:
            logger.info(f"HyDE: 实体命中（query含已知菜名）→ 跳过")
            return False

        # 无实体命中 = 知识库词汇gap → 触发HyDE
        logger.info(f"HyDE: 无实体命中，触发假设文档扩写（{len(documents)}条结果, 桥接词汇gap）")
        return True

    def generate_hyde_document(self, query: str) -> str:
        """
        生成HyDE假设文档。
        
        让LLM基于query生成一段"假设的菜谱描述"，
        这段描述会被embed后用于向量检索。
        """
        hyde_prompt = f"""作为一个烹饪知识专家，请根据以下用户的查询，
生成一段简短的、假设性的菜谱/食材知识描述（100-200字）。
这段描述用于在菜谱知识库中进行相似度检索，所以请包含尽可能多的具体菜名、食材名和烹饪术语。

用户查询：{query}

请只输出假设描述内容，不要加"假设的描述："等前缀："""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": hyde_prompt}],
                temperature=0.3,
                max_tokens=self.config.hyde_document_max_tokens,
            )
            hyde_doc = response.choices[0].message.content.strip()
            logger.info(f"HyDE假设文档生成完成: {hyde_doc[:80]}...")
            return hyde_doc

        except Exception as e:
            logger.error(f"HyDE文档生成失败: {e}")
            # 降级：返回原始query
            return query

    # ============================================================
    # 实体提取（用于实体栈维护）
    # ============================================================

    def extract_entities_from_turn(
        self,
        rewritten_query: str,
        ai_response: str
    ) -> List[str]:
        """
        从本轮对话的消解后query和AI回答中提取已知实体名。

        提取策略：
        1. 从query中提取（Neo4j全文索引验证）
        2. 从AI回答中提取——用正则匹配推荐列表中的菜名（如 "1. 菜名"、"✅ 1. 菜名"），
           然后用Neo4j精确查询验证

        Returns:
            实体名列表（按AI回答中出现顺序排列，去重后）
        """
        candidates = []  # 用list保序

        # 从AI回答提取 —— AI推荐过的菜名就是上下文实体
        # 不再从query中提取（如"今天晚餐吃什么"这种模糊query会导致Neo4j全文索引返回随机结果）
        if ai_response:
            recipe_names = self._extract_recipe_names_from_text(ai_response)
            for name in recipe_names:
                if name not in candidates:
                    candidates.append(name)

        if candidates:
            logger.info(
                f"实体提取: 本轮发现 {len(candidates)} 个实体: {candidates}"
            )
        return candidates

    def _extract_recipe_names_from_text(self, text: str) -> List[str]:
        """
        从AI回答中提取推荐菜名。
        
        结构区分策略：AI推荐菜名行的特征是"肯定号/序号 + 中文名（"，
        而属性描述行是"肯定号 + 属性标签：值（"。两者都有✅和（，
        但菜名的（前只有中文字符，属性标签的（前有：等分隔符。
        """
        seen = set()
        result = []

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # 只处理有推荐列表标记的行：✅ 或 数字.
            has_marker = (
                '✅' in line[:3] or
                bool(re.match(r'\d+[.、．]', line))
            )
            if not has_marker:
                continue

            # 清除前缀标记
            clean = re.sub(r'✅\s*', '', line)
            clean = re.sub(r'^\d+[.、．]\s*', '', clean)
            clean = re.sub(r'^\*{1,2}', '', clean)
            clean = clean.strip()

            # 找到第一个 （ 或 ( 的位置
            paren_pos = -1
            for i, c in enumerate(clean):
                if c in '（(':
                    paren_pos = i
                    break

            if paren_pos > 0:
                # 格式A：菜名（分类）— 括号前只能有中文字符
                prefix = clean[:paren_pos].strip()
                if not prefix:
                    continue
                if not re.fullmatch(r'[\u4e00-\u9fff]+', prefix):
                    # 含非中文字符（如：菜品明确标注：分类 → 冒号）→ 属性标签，跳过
                    continue
                name = prefix
            else:
                # 格式B：纯菜名独占一行（如 "✅ 1. 火腿蛋炒饭"）
                if not re.fullmatch(r'[\u4e00-\u9fff]{2,12}', clean):
                    continue
                name = clean

            if len(name) < 2 or len(name) > 12:
                continue
            if name in seen:
                continue
            if any(bad in name for bad in _RECIPE_NAME_BLACKLIST):
                continue
            seen.add(name)
            result.append(name)

        logger.debug(f"文本菜名提取: {result}")
        return result
