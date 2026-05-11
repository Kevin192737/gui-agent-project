# GUI Agent Project

一个面向移动端界面任务的 GUI Agent 基线项目，支持基于截图执行单步动作决策，并提供离线评测流程。

## Features

- 支持标准动作空间：`CLICK`、`TYPE`、`SCROLL`、`OPEN`、`COMPLETE`
- 包含任务解析、阶段规划、动作后处理与鲁棒解析逻辑
- 提供本地离线评测与可视化辅助工具
- 适合用于 GUI Agent 策略实验与基线对比

## Project Structure

- `agent.py`: 主 Agent 实现（评测入口默认加载）
- `agent_base.py`: Agent 协议、基础类与 API 调用封装
- `test_runner.py`: 本地评测脚本
- `test_data/`: 离线测试数据集
- `utils/`: 评测辅助与可视化工具
- `docs/`: 设计说明文档
- `requirements.txt`: Python 依赖列表

## Requirements

- Python 3.10+
- Windows / Linux / macOS（推荐使用虚拟环境）

## Quick Start

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

运行本地评测：

```bash
python test_runner.py
```

## Configuration

如需在线模型推理，请按运行环境设置相应密钥（例如 `VLM_API_KEY`）。  
离线评测默认读取 `test_data/` 下的数据，无需联网。

## Notes

- `test_runner.py` 在部分比赛/评测场景中可能会被平台替换，请以评测平台规则为准。
- 提交前建议检查仓库中是否包含不应公开的数据或凭据文件。
