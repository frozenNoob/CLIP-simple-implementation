# 基于CLIP实现跨模态检索

本项目基于kaggle上的一个已有的项目（[OpenAI CLIP simple implementation](https://www.kaggle.com/code/moeinshariatnia/openai-clip-simple-implementation/notebook#Dataset)）实现。我在此基础上添加了一个公共数据集（flickr8k）和一个由我队友们设计的大小为300的数据集。并把损失函数由$InfoNCE$改为$ InfoNCE+\alpha \times TripletMarginLoss$。

需要注意的是，flickr30k是flickr8k的扩展（图片部分重复，但是标注是不同的）。

## 相关文档

> - [‍‌﻿‬‬⁠⁠⁠‍⁠‌‬⁠‍‍‬‍‍‌‍⁠‬‍⁠﻿‍‍⁠⁠﻿‬多模态和跨模态 - 飞书云文档](https://fcneheqzlq8n.feishu.cn/wiki/KIYGwU9SVin6sfk0DducAx9inZc)

## 环境配置

### 用kaggle下载数据集

> - https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

1. 导入环境变量(覆盖kaggle默认的存储位置)

```bash
export KAGGLEHUB_CACHE="./autodl-tmp"
echo $KAGGLEHUB_CACHE
```

2. 执行下面的python脚本（随便放到一个.py文件然后执行即可）：

```python
import kagglehub
# 下载数据集
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
print("Path to dataset  Flickr 30k:", path)  #
```

### 用HuggingFace下载模型

> - [怎么在AutoDL上面使用HuggingFace（亲测有效）_autodl huggingface-CSDN博客](https://blog.csdn.net/qq_60735796/article/details/145406385)

下载模型到本地然后加载，由此避免国内访问HF的超时问题。

```bash
# 安装 huggingface_hub 工具包
pip install huggingface_hub

# 设置镜像网站
export HF_ENDPOINT=https://hf-mirror.com

# 下载 distilbert-base-uncased 模型到本地目录
huggingface-cli download distilbert-base-uncased \
   --resume-download \
   --local-dir ./autodl-tmp/model/distilbert-base-uncased \

# 下载resnet50 模型到本地目录
huggingface-cli download timm/resnet50.a1_in1k \
  --local-dir ./autodl-tmp/model/resnet50

```



## 实验结果

部分实验条件：

- $Epoch=30$
- $训练集:测试集 = 4:1$
- $ Batch\ size = 32$

### 结果1

下面的是在测试集上得出的损失值。

| Dataset                | $ InfoNCE\ Loss$ | $InfoNCE\ Loss +\alpha \times TripletMargin\ Loss$ |
| ---------------------- | ---------------- | -------------------------------------------------- |
| flickr30k              |                  |                                                    |
| flickr8k               | 2.2464           | 2.2891                                             |
| ourDataset(nearly 300) |                  |                                                    |

### 结果2

下面是图片展示：

