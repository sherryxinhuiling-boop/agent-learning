# agent-learning

sherry 的 AI Agent 学习项目 —— 16 周，从零代码基础到独立开发 Agent 产品。

## 这是什么

这个仓库记录我从零开始学习 AI Agent 开发的全过程：每一阶段的练习脚本、项目代码和学习笔记。

## 环境

| 项目 | 版本 / 位置 |
|---|---|
| Python | 3.13.16（`/usr/local/bin/python3`） |
| 虚拟环境 | `.venv/`（不提交到仓库，见 `.gitignore`） |
| 编辑器 | VS Code |

## 怎么运行

```bash
# 1. 进入项目目录
cd ~/Projects/agent-learning

# 2. 激活虚拟环境（每天开工第一件事，提示符出现 (.venv) 才算成功）
source .venv/bin/activate

# 3. 运行某个脚本
python 01_basics.py
```

## 文件说明

| 文件 | 内容 |
|---|---|
| `llm.py` | **大模型调用模块**（可复用）。封装请求、错误处理、成本估算、流式输出、多轮对话 |
| `01_basics.py` | Python 六个核心概念：变量、f-string、列表与循环、条件判断、字典、函数 |
| `02_file_organizer.py` | 项目 0.1：文件整理脚本，按扩展名分类复制文件（只复制，不移动不删除） |
| `03_api_call.py` | 项目 0.2：无框架的原始 HTTP 调用，揭开「调用大模型」的黑箱 |
| `04_prompt_engineering.py` | 阶段 1：Prompt 对比实验，同一任务的五种写法 + 两层判定标准 |
| `05_reliable_extract.py` | 阶段 1：可靠抽取函数，Prompt + JSON 模式 + 校验重试三层防护 |
| `.env` | API 密钥，**已被忽略，不入库** |
| `sandbox/` | `02_file_organizer.py` 的演示沙盒，可随时删除重建 |

## 学习进度

- [x] 阶段 0 · 开发环境搭建
- [x] 阶段 0 · 第一个 Python 脚本（6 个概念 + 5 个练习）
- [x] 阶段 0 · 项目 0.1 文件整理脚本
- [x] 阶段 0 · 项目 0.2 API 调用脚本
- [x] 阶段 0 · 推送到 GitHub
- [ ] 阶段 1 · LLM 应用入门
- [ ] 阶段 2 · RAG 知识库
- [ ] 阶段 3 · Agent 核心
- [ ] 阶段 4 · 产品化上线
