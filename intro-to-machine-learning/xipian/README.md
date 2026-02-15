# xipian - Kaggle机器学习学习项目

这是一个使用 [uv](https://github.com/astral-sh/uv) 管理的Python项目，用于学习和实践Kaggle上的机器学习课程。

## 📋 项目概述

- **项目名称**: xipian
- **版本**: 0.1.0
- **Python版本**: >=3.12
- **包管理器**: uv
- **目的**: 学习Kaggle上的机器学习课程，实践数据科学和机器学习技能

## 🚀 快速开始

### 环境准备

1. 确保已安装Python 3.12或更高版本
2. 安装uv包管理器（如果尚未安装）：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### 安装依赖

使用uv安装项目依赖：

```bash
# 创建虚拟环境并安装依赖
uv sync
```

### 激活虚拟环境

```bash
# 激活虚拟环境
source .venv/bin/activate
```

### 运行项目

```bash
# 运行主程序
python main.py

# 运行测试脚本
python src/test.py
```

## 📁 项目结构(动态更新)

```
xipian/
├── .python-version      # Python版本指定
├── .venv/               # 虚拟环境目录（由uv自动创建）
├── exp/                 # 实验和探索性分析目录
│   └── melb_data.ipynb  # Melbourne房价数据探索
├── input/               # 数据集目录
│   └── melb_data.csv    # Melbourne房价数据
├── src/                 # 源代码目录
│   └── test.py          # 测试脚本
├── main.py              # 项目主入口文件
├── pyproject.toml       # 项目配置和依赖管理
├── README.md            # 项目说明文档
└── uv.lock              # uv锁文件，确保依赖一致性
```

## 📦 依赖管理

项目使用uv进行依赖管理，主要依赖包括：

- **kaggle** (>=2.0.0): Kaggle API客户端，用于下载数据集和提交结果
- **pandas** (>=3.0.0): 数据处理和分析库

### 添加新依赖

```bash
# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name
```

### 更新依赖

```bash
# 更新所有依赖
uv sync
```

## 🎯 学习目标

本项目旨在通过Kaggle课程学习以下内容：

1. **数据预处理和探索性数据分析**
2. **机器学习模型构建和评估**
3. **特征工程和模型优化**
4. **Kaggle竞赛参与和提交**

## 📚 学习资源

- [Kaggle学习平台](https://www.kaggle.com/learn)

## 🔧 开发指南

### 代码规范

- 遵循PEP 8 Python代码风格指南
- 使用有意义的变量和函数名
- 添加适当的注释和文档字符串

### 虚拟环境管理

项目使用uv管理的虚拟环境，位于`.venv`目录。建议不要将此目录提交到版本控制。

### 版本控制

项目使用Git进行版本控制，遵循常规的Git工作流程。

## 🤝 贡献指南

欢迎对项目进行改进和扩展！请遵循以下步骤：

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 🙏 致谢

- 感谢Kaggle提供优质的学习资源和平台
- 感谢uv团队开发了优秀的Python包管理器
- 感谢所有开源社区的贡献者

---

**开始你的机器学习之旅吧！** 🚀
