# GUI Agent Project

这是从你当前目录中整理出的可发布版本，已按 GitHub 项目方式组织，便于版本管理与协作。

## 项目结构

- `agent.py`：主 Agent 逻辑入口
- `agent_base.py`：基础能力与协议定义
- `test_runner.py`：本地评测入口
- `test_data/`：测试数据
- `utils/`：辅助工具
- `docs/`：说明文档
- `requirements.txt`：Python 依赖

## 环境准备

建议 Python 3.10+

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 本地运行

```bash
python test_runner.py
```

如果需要在线模型调用，请按你原有方式设置环境变量（例如 `VLM_API_KEY`）。

## 发布到 GitHub

在项目目录下执行：

```bash
git init
git add .
git commit -m "init gui agent project"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```
