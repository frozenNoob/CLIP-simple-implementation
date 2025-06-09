# 基于CLIP实现跨模态检索

本项目基于kaggle上的一个已有的项目（[OpenAI CLIP simple implementation](https://www.kaggle.com/code/moeinshariatnia/openai-clip-simple-implementation/notebook#Dataset)）实现。我在此基础上添加了一个公共数据集（flickr8k）和一个由我队友们设计的大小为300的数据集。并把损失函数由$InfoNCE$改为$ InfoNCE+\alpha \times TripletMarginLoss$。

需要注意的是，flickr30k是flickr8k的扩展（图片部分重复，但是标注是不同的）。

## 相关文档

> - [‍‌﻿‬‬⁠⁠⁠‍⁠‌‬⁠‍‍‬‍‍‌‍⁠‬‍⁠﻿‍‍⁠⁠﻿‬多模态和跨模态 - 飞书云文档](https://fcneheqzlq8n.feishu.cn/wiki/KIYGwU9SVin6sfk0DducAx9inZc)

## 原理分析

### 1） 加入三元组损失

> - [TripletMarginLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html)

**三元组损失**【这里我选取的正样本是样本对应的标签，比如在训练集中的image样本对应的Text样本】：

![img](https://fcneheqzlq8n.feishu.cn/space/api/box/stream/download/asynccode/?code=YmQ1MzA2OTIwOWIxYmM2ZDY1NTdmMWRmNGZjMmNhOWRfVFVIN2JZdmNNNENyN0ZNWXRYUVhjMWlNZzk2b1hzRUdfVG9rZW46RDFWQWJyOUxSb0pZNER4aVJLUGNZcE15bmdjXzE3NDk0OTA0ODA6MTc0OTQ5NDA4MF9WNA)

- **为何融合 InfoNCE 能改进 CLIP** 

  CLIP 原版对比学习用的 InfoNCE 损失形式为：

![img](https://fcneheqzlq8n.feishu.cn/space/api/box/stream/download/asynccode/?code=MmIzOWUwNGJlMTY4OTBkZTY0ZGQ2NGNmZGYyMWVhMDlfcHcyUVVpcUFPdjc2OW5tcHNRRkNCVENia2dvcTdkbElfVG9rZW46UkhKQmJpQnRub2tnUlp4R3hKOGNaQjRnbjJmXzE3NDk0OTA0ODA6MTc0OTQ5NDA4MF9WNA)

或者下面那种（**更方便理解**：[InfoNCE Loss公式及源码理解-CSDN博客](https://blog.csdn.net/weixin_43427721/article/details/134539003)）：

![img](https://fcneheqzlq8n.feishu.cn/space/api/box/stream/download/asynccode/?code=MmRmYzgzMDE2NTBhZGU0OTE2OTQzZGRjMDc4NjFjMjBfN3l4V2hyMGNFWXlVSm9wZWNtSWkxVG9tMlV3Y0RlU3JfVG9rZW46RjZPZGJKendlb3FZMEZ4S0dLNGN2N05DbkRnXzE3NDk0OTA0ODA6MTc0OTQ5NDA4MF9WNA)

- 这里它同时把每个正对 (xi,yi)(x_i,y_i)(xi,yi) 与**所有**其他负对拉开。InfoNCE 的优点是批内全负样本一次考虑，保证全局分布对齐；但缺点也明显：
  - **弱化最难负样本**：对所有负样本一视同仁，hard negative（最具迷惑性的负样本）得到的梯度信号比较平均，不够突出；
  - **只关注相似度，不直接控制“距离 margin”**：没有显式 margin，将相似度推低的“安全边界”是隐含的。
- 三元组损失则：
  - **显式控制 margin**：保证每个锚点到正样本的距离，至少比到负样本距离小 margin；
  - **聚焦 hardest negative**：如果采样 hardest 或 semi-hard 负样本，会给最具挑战性的负对更大梯度，能更快收敛到更区分度高的嵌入空间。
- **融合的思路**

![img](https://fcneheqzlq8n.feishu.cn/space/api/box/stream/download/asynccode/?code=MmRmYjRlNjE5MTgyZDExZjQxMjUyNzNiMDhkOWIxNzdfSUlBd3JaZGptRW44UFFFRlNVNUZkMVVzYnliYXRqWWRfVG9rZW46T0w4bmJiYW4ybzFMSWx4S3lWVmMxRWVkbmhiXzE3NDk0OTA0ODA6MTc0OTQ5NDA4MF9WNA)

- **全局＋局部**：InfoNCE 保证批内全局语义对齐，Triplet Loss 强化 hardest negative 的 margin 分离；
- **互补提升**：InfoNCE 平滑分布式对比，Triplet Loss 明确 margin 约束，二者结合能让嵌入空间既整体一致，又局部更具判别力；
- **超参**可调：$ \alpha$ 控制二者权重，可在验证集上调优，找到性能最优的平衡。

这样，融合后既保留了 CLIP 原有的高效全负样本对比，又通过三元组的硬负样本挖掘和 margin 强化，实现了更鲁棒、更具区分度的视觉–语言表示学习。

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
- $ \alpha = 0.1$

在GTX 3090显卡上，对数据集flickr30k 训练用时为4h 47min 38s。

### 结果1

下面的是在测试集上得出的损失值。

| Dataset                  | $ InfoNCE\ Loss$ | $InfoNCE\ Loss +\alpha \times TripletMargin\ Loss$ |
| ------------------------ | ---------------- | -------------------------------------------------- |
| $flickr30k$              | $2.1012$         | $2.2114(=2.1861+0.0253)$                           |
| $flickr8k$               | $2.2464$         | $2.2891(=2.2646+0.0245)$                           |
| $ourDataset(nearly 300)$ | $3.3636$         | $3.3846(=3.3635+0.0211)$                           |

但是实际上前2个数据集的表现并不理想。

### 结果2

图例中的$one\ loss$是指$ InfoNCE\ Loss$， 而$ two\ loss$是指  $InfoNCE\ Loss +\alpha \times TripletMargin\ Loss$。

下面是图片展示：

![image-20250610013036353](./assets/image-20250610013036353.png)

![image-20250610014149276](./assets/image-20250610014149276.png)

![image-20250610013324766](./assets/image-20250610013324766.png)

### 结果3

![image-20250610021630698](./assets/image-20250610021630698.png)

![image-20250610021305529](./assets/image-20250610021305529.png)







![image-20250610021915433](./assets/image-20250610021915433.png)

![image-20250610022154914](./assets/image-20250610022154914.png)

![image-20250610022637355](./assets/image-20250610022637355.png)

![image-20250610023022731](C:\Users\WB\AppData\Roaming\Typora\typora-user-images\image-20250610023022731.png)
