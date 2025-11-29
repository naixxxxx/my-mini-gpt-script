# my-mini-gpt-script

一个「从零写到能跑起来」的小型 GPT 语言模型脚本仓库  
代码主要来自我自己学习 Stanford CS336 / Karpathy NanoGPT 时的实现和精简，把训练和推理需要的脚本单独抽出来，方便在本地 GPU（比如一张 RTX 3060）上快速复现一个可用的 mini-GPT。




.
├── cs336_basics/
│   ├── TransformerLM.py          # 模型定义
│   ├── ...                       # RMSNorm / RoPE / Attention / FFN 等模块
│   └── pipeline/
│       ├── trainmodel.py         # 训练脚本（主入口）
│       └── inference.py          # 推理 / 生成脚本（在终端里对话）
