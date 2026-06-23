# 今天吃什么 - AI美食推荐助手

本项目是基于 [Datawhale/all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程进行二次开发的完整实战项目，在原始 RAG 教学框架基础上，新增了多项生产级增强功能，构建了一个智能烹饪助手系统。

![界面](./view.png)

## 项目说明

### 与 all-in-rag 的关系

本项目以 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 为基础框架，保留了其核心的图 RAG 架构（Neo4j 图数据库 + Milvus 向量数据库 + 混合检索），在此基础上进行了大量二次开发，新增了全链路追踪、质量评估、查询改写、重排序等生产级功能，使系统从教学 Demo 升级为可实际使用的智能问答系统。

**原始框架提供的基础能力：**
- Neo4j 图数据库建模与查询
- Milvus 向量数据库索引与检索
- 基础 RAG 流程（数据准备 → 索引构建 → 检索 → 生成）
- Docker 一键部署方案
- 前端交互界面

**本项目新增的增强功能（详见下文）：**
- Langfuse 全链路追踪
- RAGAS 质量评估体系
- CrossEncoder 重排序（三阶段漏斗精排）
- 指代消解（LLM + 规则降级）
- HyDE 假设文档扩写
- BM25 全文检索
- Parent-Child 文档切分
- 三级递进式查询引擎

## 新增功能详解

### 1. Langfuse 全链路追踪

**模块文件：** `rag_modules/observability.py`

集成 [Langfuse](https://langfuse.com/) 实现 RAG 全链路可观测性，覆盖从用户提问到答案生成的完整流程：

- **Trace 追踪**：每次用户查询创建独立 Trace，记录完整处理链路
- **Span 分段**：将处理流程拆分为 `cache_check`、`query_rewrite`、`retrieval`、`hyde_expansion`、`answer_generation`、`post_process` 等 Span，可精确定位性能瓶颈
- **Score 评分**：支持用户反馈和 RAGAS 评估分数回写到 Trace，实现质量闭环
- **自动降级**：Langfuse 未配置或连接失败时自动降级为 no-op，不影响业务逻辑

### 2. RAGAS 质量评估体系

**模块文件：** `rag_modules/ragas_eval.py`、`scripts/batch_ragas_eval.py`

采用「在线收集 + 离线评估」的解耦架构：

- **数据收集**（`rag_modules/ragas_eval.py`）：在每次问答时自动收集 `{question, contexts, answer, trace_id}` 写入 JSONL 文件
- **批量评估**（`scripts/batch_ragas_eval.py`）：在独立 conda 环境中运行 RAGAS 评估，支持 `faithfulness`（忠实度）和 `context_precision`（上下文精确度）指标
- **分数回写**：评估完成后自动将分数按 `trace_id` 回写到 Langfuse，可在网页端直接查看

### 3. CrossEncoder 重排序（三阶段漏斗精排）

**模块文件：** `rag_modules/reranker.py`

实现三阶段漏斗式精排策略，兼顾精度和速度：

| 阶段 | 方法 | 模型 | 耗时 | 说明 |
|------|------|------|------|------|
| A | Embedding 余弦粗筛 | bge-small-zh-v1.5（已加载） | ~0.3s | 复用已有 embedding，零额外加载成本 |
| B | Cross-Encoder 精排 | bge-reranker-base | ~1-2s | 对通过粗筛的候选深度打分 |
| C | 输出 top_k | - | - | 取精排后最高分结果 |

**智能优化：** 简单查询（复杂度 < 0.3）自动跳过 Cross-Encoder，仅用 Embedding 余弦排序，响应时间 < 0.5s。

### 4. 指代消解

**模块文件：** `rag_modules/query_rewriter.py`

解决多轮对话中的指代歧义问题（如「第一个怎么做」「它和那个比哪个更辣」）：

- **LLM 优先路径**：检测到指代词后，构建滑动窗口上下文（默认最近 3 轮对话），通过精心设计的 prompt（含 5 个 few-shot 示例）调用 LLM 进行消解
- **规则降级路径**：LLM 消解失败时，基于规则引擎从实体栈中映射指代对象，支持序数指代（第一个/第二个）、前文指代（刚才说的）、替代指代（换一个）等多种模式
- **实体栈管理**：通过 `SessionCacheManager` 维护每轮对话的实体栈，为指代消解提供上下文支撑

### 5. HyDE 假设文档扩写

**模块文件：** `rag_modules/query_rewriter.py`

当检索结果不足时，自动生成假设文档增强检索：

- **触发条件**：检索结果数量少于阈值（默认 3 条）或最高分低于阈值（默认 0.4），且无 Neo4j 实体命中
- **工作流程**：LLM 生成假设文档 → Embedding 向量化 → 向量检索 → 与原始结果合并去重
- **实体命中检测**：先查 Neo4j 确认 query 中是否包含已知菜名/食材名，有实体命中则跳过 HyDE（直接检索通常已足够）

### 6. BM25 全文检索

**模块文件：** `rag_modules/hybrid_retrieval.py`

集成 `rank-bm25` 实现关键词精确匹配检索，作为向量检索和图检索的补充：

- 使用 Child 块（Parent-Child 切分策略中的小块）构建 BM25 索引
- 检索命中 Child 块后自动扩展为 Parent 块返回（上下文更完整）
- 与向量检索、图检索结果合并，形成多路召回

### 7. Parent-Child 文档切分

**模块文件：** `rag_modules/graph_data_preparation.py`、`rag_modules/hybrid_retrieval.py`

采用分层切分策略，平衡检索精度和上下文完整性：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `parent_chunk_size` | 1000 字符 | Parent 块大小，用于最终返回给 LLM |
| `child_chunk_size` | 250 字符 | Child 块大小，用于向量检索和 BM25 检索 |
| `child_chunk_overlap` | 30 字符 | Child 块重叠，避免语义截断 |

**工作原理：** Child 块用于精确匹配和向量检索（精度高），命中后通过 `parent_id` 关联取回 Parent 块返回给 LLM（上下文更完整）。

### 8. 三级递进式查询引擎

**模块文件：** `rag_modules/intelligent_query_router.py`

实现智能查询路由，根据查询复杂度自动选择最优检索策略：

**查询分析（规则优先 → LLM 兜底）：**
- **第一层**：精确模式匹配（正则），置信度 0.85-0.92，覆盖因果推理、对比分析、多实体关系、直接菜谱查询等常见句式
- **第二层**：关键词加权评分，推理/关系/过程关键词分级计分
- **第三层**：LLM 兜底分析（可选），仅在规则置信度不足时调用

**三级检索策略：**

| 策略 | 适用场景 | 检索引擎 |
|------|----------|----------|
| `hybrid_traditional` | 简单信息查找（怎么做、推荐菜） | 向量检索 + BM25 + 图索引 |
| `graph_rag` | 复杂关系推理（为什么、对比分析） | 图 RAG 多跳遍历 |
| `combined` | 综合查询（条件推荐、特征分析） | 传统检索 + 图检索合并 |

**按需重排：** 简单查询跳过 Cross-Encoder（轻量 embedding 排序），复杂查询启用漏斗精排。

## 系统架构

```
用户提问
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                  Langfuse 全链路追踪                   │
│  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ 语义缓存 │→│ 指代消解  │→│  智能查询路由       │   │
│  │ 检查     │  │ (LLM+规则)│  │  (规则→LLM兜底)   │   │
│  └─────────┘  └──────────┘  └───────┬───────────┘   │
│                                      │               │
│              ┌───────────────────────┼───────┐       │
│              ▼                       ▼       ▼       │
│     ┌──────────────┐  ┌──────────┐  ┌──────────┐   │
│     │ 传统混合检索  │  │ 图RAG检索│  │ BM25检索 │   │
│     │ (向量+图索引) │  │ (多跳遍历)│  │ (关键词) │   │
│     └──────┬───────┘  └────┬─────┘  └────┬─────┘   │
│            └───────────────┼──────────────┘         │
│                            ▼                        │
│              ┌──────────────────────────┐           │
│              │   CrossEncoder 重排序     │           │
│              │ (Embedding粗筛→CE精排)    │           │
│              └────────────┬─────────────┘           │
│                           ▼                         │
│              ┌──────────────────────────┐           │
│              │   HyDE 假设文档扩写       │           │
│              │  (检索不足时触发)         │           │
│              └────────────┬─────────────┘           │
│                           ▼                         │
│              ┌──────────────────────────┐           │
│              │      LLM 答案生成         │           │
│              └────────────┬─────────────┘           │
│                           ▼                         │
│              ┌──────────────────────────┐           │
│              │   RAGAS 数据收集          │           │
│              │  (question→contexts→answer)│          │
│              └──────────────────────────┘           │
└─────────────────────────────────────────────────────┘
  │
  ▼
返回答案
```

## 特性

- **智能推荐**：基于图 RAG + 向量检索 + BM25 的多路召回 AI 推荐
- **全链路可观测**：Langfuse 追踪每个处理环节，支持质量评分回写
- **多轮对话理解**：指代消解 + 实体栈管理，准确理解上下文指代
- **自适应检索**：智能路由自动选择最优策略，简单查询快速响应，复杂查询深度推理
- **检索质量保障**：CrossEncoder 重排序 + HyDE 假设文档扩写，确保检索结果相关
- **详细指导**：分步骤烹饪指南，新手也能轻松上手
- **现代界面**：玻璃质感的响应式设计，支持桌面和移动设备

## 快速开始

### 前置要求

- **Docker Desktop** - [下载安装](https://www.docker.com/products/docker-desktop/)
- **Node.js 18+** - [下载安装](https://nodejs.org/) (可选，仅前端开发时需要)
- **Python 3.11+** - 后端运行环境
- **Conda** (可选) - 用于 RAGAS 评估的独立环境

### 环境配置

**1. 克隆项目**

```bash
git clone https://github.com/FutureUnreal/What-to-eat-today.git
cd What-to-eat-today
```

**2. 配置环境变量**

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
```

需要配置的关键环境变量：

| 变量名 | 说明 | 必填 |
|--------|------|------|
| `OPENAI_API_KEY` | LLM API 密钥（支持所有 OpenAI 兼容格式） | 是 |
| `OPENAI_BASE_URL` | LLM API 地址 | 是 |
| `LLM_MODEL` | LLM 模型名称 | 是 |
| `EMBEDDING_MODEL` | Embedding 模型名称 | 否（默认 bge-small-zh-v1.5） |
| `NEO4J_PASSWORD` | Neo4j 数据库密码 | 否（默认 all-in-rag） |
| `LANGFUSE_HOST` | Langfuse 服务地址 | 否（未配置则自动禁用） |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 项目公钥 | 否 |
| `LANGFUSE_SECRET_KEY` | Langfuse 项目密钥 | 否 |

### 一键启动（Docker）

**Windows 用户：**
```bash
start.bat
```

**Linux/macOS 用户：**
```bash
chmod +x start.sh stop.sh
./start.sh
```

### 访问应用

启动完成后，访问：

- **应用首页**：http://localhost
- **前端**：http://localhost:3000
- **后端 API**：http://localhost:8000
- **Neo4j**：http://localhost:7474 (neo4j/all-in-rag)
- **Milvus**：http://localhost:9001 (minioadmin/minioadmin)

## 依赖安装

### 主环境依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含所有核心依赖：

| 依赖包 | 用途 |
|--------|------|
| `transformers` + `sentence-transformers` | Embedding 模型 & CrossEncoder 重排序 |
| `langchain-core` / `langchain-community` / `langchain-huggingface` / `langchain-text-splitters` | RAG 框架 |
| `neo4j` | Neo4j 图数据库连接 |
| `pymilvus` | Milvus 向量数据库连接 |
| `rank-bm25` | BM25 全文检索 |
| `openai` | LLM API 调用 |
| `langfuse` | 全链路追踪 |
| `flask` + `flask-cors` | Web 服务 |
| `tiktoken` | Token 计数 |
| `numpy` / `pandas` / `scikit-learn` / `scipy` | 数据处理 & 余弦计算 |

### RAGAS 评估环境依赖（可选）

RAGAS 与项目主环境存在依赖冲突，需在独立 conda 环境中运行：

```bash
# 创建独立的评估环境
conda create -n ragas-eval python=3.11 -y
conda activate ragas-eval

# 安装 RAGAS 及相关依赖
pip install ragas datasets langchain-openai

# 安装 Langfuse（用于分数回写）
pip install langfuse

# 安装 dotenv（用于加载 .env 配置）
pip install python-dotenv

# 运行批量评估
python scripts/batch_ragas_eval.py

# 退出评估环境
conda deactivate
```

## 停止服务

```bash
# 停止所有服务
stop.bat        # Windows
./stop.sh       # Linux/macOS

# 或者直接使用 Docker Compose
docker-compose down

# 完全重置（清除所有数据）
docker-compose down -v
```

## 常用命令

### Docker 服务管理

```bash
# 查看所有服务状态
docker-compose ps

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f neo4j

# 重启特定服务
docker-compose restart backend

# 进入后端容器
docker exec -it what-to-eat-backend bash

# 进入 Neo4j 容器执行 Cypher 查询
docker exec -it what-to-eat-neo4j cypher-shell -u neo4j -p all-in-rag

# 完全重置（清除所有数据）
docker-compose down -v
```

### RAGAS 评估

```bash
# 激活评估环境
conda activate ragas-eval

# 运行批量评估（读取 data/ragas_batch.jsonl，评分后写回 Langfuse）
python scripts/batch_ragas_eval.py

# 退出评估环境
conda deactivate
```

### 前端开发

```bash
# 本地运行前端开发服务器（需先停止容器化前端）
docker-compose stop frontend
cd frontend
npm install
npm run dev
# 访问 http://localhost:3000

# 构建前端生产版本
cd frontend
npm run build
```

### 后端开发

```bash
# 本地运行后端（需先停止容器化后端）
docker-compose stop backend
pip install -r requirements.txt
python main.py
# 访问 http://localhost:8000
```

### 数据库操作

```bash
# 查看 Neo4j 节点统计
docker exec -it what-to-eat-neo4j cypher-shell -u neo4j -p all-in-rag "MATCH (n) RETURN labels(n) AS type, count(n) AS count ORDER BY count DESC"

# 查看 Neo4j 关系统计
docker exec -it what-to-eat-neo4j cypher-shell -u neo4j -p all-in-rag "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"

# 查看 Milvus 集合信息
docker exec -it what-to-eat-milvus curl http://localhost:19530/v2/vectordb/collections/list
```

## 技术栈

**前端：**
- Next.js 14 (React 框架)
- Tailwind CSS (样式框架)
- Zustand (状态管理)
- Framer Motion (动画)

**后端：**
- Python 3.11 + Flask
- Neo4j (图数据库)
- Milvus (向量数据库)
- LangChain (RAG 框架)
- Sentence-Transformers (Embedding & CrossEncoder)
- Langfuse (全链路追踪)
- RAGAS (质量评估)
- rank-bm25 (全文检索)

**部署：**
- Docker + Docker Compose
- Nginx (反向代理)

## 故障排除

### 问题1：Docker 未运行
```
错误：Docker 未运行
解决：启动 Docker Desktop
```

### 问题2：端口被占用
```
错误：端口 80/3000/8000 被占用
解决：关闭占用端口的程序，或修改 docker-compose.yml 中的端口映射
```

### 问题3：服务启动超时
```
错误：服务启动超时
解决：
1. 检查网络连接
2. 查看具体服务日志
3. 重启 Docker Desktop
4. 清理 Docker 缓存：docker system prune -f
```

### 问题4：Langfuse 连接失败
```
提示：Langfuse 连接失败，追踪已禁用
说明：这是正常行为，Langfuse 为可选功能，未配置时自动降级，不影响系统运行
解决：检查 .env 中 LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 是否正确
```

### 问题5：CrossEncoder 模型加载失败
```
提示：重排序模型加载失败，降级为 embedding 排序
说明：系统会自动降级，不影响基本功能
解决：确保 HuggingFace 模型可访问，或配置 HF_ENDPOINT 镜像地址
```

## 致谢

本项目的开发得益于以下开源项目和教程：

### 教程项目
- **[Datawhale/all-in-rag](https://github.com/datawhalechina/all-in-rag)** - 大模型应用开发实战：RAG 技术全栈指南
  - 本项目是该教程的完整实战案例，在其基础上进行了大量二次开发
  - 原始框架提供了 Neo4j 图数据库建模、Milvus 向量检索、Docker 部署等基础能力
  - 本项目在此基础上新增了 Langfuse 追踪、RAGAS 评估、CrossEncoder 重排序、指代消解、HyDE、BM25、Parent-Child 切分、智能查询路由等生产级功能

### 菜谱数据
- **[Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)** - 程序员在家做饭方法指南
  - 本项目的菜谱数据主要来源于这个优秀的开源项目

### 技术栈
- **Langfuse** - 开源 LLM 可观测性平台
- **RAGAS** - RAG 系统评估框架
- **Sentence-Transformers** - 文本嵌入和重排序模型
- **Next.js** - React 全栈框架
- **Flask** - Python Web 框架
- **Neo4j** - 图数据库
- **Milvus** - 向量数据库
- **Docker** - 容器化部署
- **Tailwind CSS** - 样式框架

**享受您的美食推荐之旅！** 如果这个项目对您有帮助，请给个 Star 支持一下！

## 许可证

本项目采用 [MIT License](./LICENSE) 开源协议。

### 使用权限
- 商业使用 - 可以用于商业项目
- 修改 - 可以修改源代码
- 分发 - 可以分发原始或修改后的代码
- 私人使用 - 可以私人使用
- 专利使用 - 授予专利使用权

### 使用条件
- **保留版权声明** - 在所有副本中包含原始版权声明和许可证声明
- **保留许可证** - 在所有副本中包含 MIT 许可证

### 免责声明
- 软件按"原样"提供，不提供任何明示或暗示的保证
- 作者不承担任何责任或义务

---

Copyright (c) 2025 FutureUnreal. All rights reserved.
