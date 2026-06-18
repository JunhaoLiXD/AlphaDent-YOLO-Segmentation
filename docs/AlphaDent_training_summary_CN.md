# AlphaDent YOLO 分割训练总结

本文档总结目前已经完成的 AlphaDent YOLO segmentation 训练实验。  
主要目的是记录每一版代码修改了什么参数、为什么这样修改、修改之后结果变好还是变差，以及可以从中得到什么结论。

在这个项目中，最重要的开发指标是 **Mask mAP50-95**。  
因为这是一个分割任务，严格的 mask 质量比只大概检测到目标区域更重要。

---

## 1. 整体实验时间线

| 版本 | 主要目的 | 模型 | 图片尺寸 | 主要修改 | 最佳 Mask mAP50-95 | 结果 |
|---|---|---:|---:|---|---:|---|
| V5 | 低分辨率 YOLOv8s baseline | YOLOv8s-seg | 640 | 第一个稳定 baseline | 0.1975 | 有效 baseline |
| V6 | 测试更高分辨率 | YOLOv8s-seg | 768 | 将 `imgsz` 从 640 提高到 768 | 0.2336 | 提升，目前最佳 baseline |
| V7 | 测试高分辨率 + rare Caries 策略 | YOLOv8s-seg | 896 | 提高 `imgsz`，rare Caries oversampling，减弱强 augmentation | 0.2260 | 低于 V6 |
| V8 | 单独测试图片尺寸 | YOLOv8s-seg | 896 | 只提高 `imgsz` 到 896，不使用 oversampling 和额外 augmentation 修改 | 0.2260 | 低于 V6 |
| V9 | 测试更大模型容量 | YOLOv8m-seg | 768 | 将模型从 YOLOv8s 换成 YOLOv8m，保持 `imgsz=768` | 0.2320 | 与 V6 接近，但没有超过 |
| V10 | 测试轻度 rare Caries oversampling | YOLOv8s-seg | 768 | YOLOv8s-seg + imgsz=768 + 轻度 rare Caries oversampling | 0.2341 | 略高于 V6 但在噪声范围内；recall 提升，precision 下降 |
| V11 | 方案 D：解耦破坏性增强 + 医疗级 copy-paste | YOLOv8s-seg | 768 | `mosaic=0`、`mixup=0`、`copy_paste=0.2` | 0.2135 | **明显回退（−0.020）**；关闭 mosaic 加剧了过拟合 |
| V12 | 从架构层面攻击小目标瓶颈 | YOLOv8s-seg + P2 头 | 768 | 加 stride-4（P2）分割头；增强回退到干净的 V6 baseline | 0.2215* | **未超过 baseline（≈−0.013）**；recall 没有提升，说明 P2 没多检出小病灶 |
| V13 | 从输入层面攻击瓶颈 | YOLOv8s-seg（tile） | 640/tile | crop / tile 裁剪训练：把每张图切成重叠 tile，在 tile 上训练 | 0.0993† | **严重退步（−0.11）**；切块把贡献大部分 mAP 的大目标切碎/丢弃 |

\* V12 的 0.2215 是 ep32 的单轮尖峰；稳定水平约 0.21，即相对 V6/V10 baseline ≈−0.02。

† V13 的 0.0993 是**同口径**的全图（切块+合并）Mask mAP50-95，对比对象是用**同一套代码**重评的 V6（0.2099），而非历史的 0.234。详见 V13 小节。

目前最好的实用 baseline（V10 技术上最高，但相比 V6 的提升可忽略不计；V11/V12/V13 都没能超过）：

```text
YOLOv8s-seg + imgsz=768 + 轻度 rare Caries oversampling
```

---

## 2. 按版本分析

## V5：YOLOv8s-seg，imgsz=640

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 640 |
| Batch size | 16 |
| Epochs | 120 |
| Patience | 25 |
| 水平翻转 | `fliplr=0` |
| 主要目的 | 建立一个稳定的低分辨率 baseline |

### 为什么这样设置

这是第一个稳定的 YOLO segmentation baseline。  
`640` 是 YOLO 训练中比较常见的起始尺寸，显存压力小，适合快速验证流程是否能跑通。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch | 27 |
| Mask Precision | 0.5641 |
| Mask Recall | 0.3798 |
| Mask mAP50 | 0.3589 |
| Mask mAP50-95 | 0.1975 |
| Box Precision | 0.5770 |
| Box Recall | 0.3857 |
| Box mAP50 | 0.3705 |
| Box mAP50-95 | 0.2268 |

### 结果理解

模型确实学到了一些目标模式，但是分割质量还比较弱。  
Mask mAP50-95 较低，说明模型可能可以大概定位一些目标区域，但严格的 mask 匹配质量还不够好。

### 从 V5 得到的结论

训练流程是有效的，但 `imgsz=640` 对这个任务可能偏小。  
因为 AlphaDent 中很多目标，尤其是 Caries 区域，非常小，所以提高图片尺寸是合理的下一步。

---

## V6：YOLOv8s-seg，imgsz=768

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 768 |
| Batch size | 16 |
| Epochs | 120 |
| Patience | 25 |
| 主要目的 | 测试更高分辨率是否能改善小目标分割 |

### 为什么修改这个参数

V5 的结果说明模型可能缺少足够的空间细节来分割较小的牙齿病灶区域。  
因此把图片尺寸提高到 `768`，希望保留更多局部信息。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch | 32 |
| Mask Precision | 0.6977 |
| Mask Recall | 0.4053 |
| Mask mAP50 | 0.4125 |
| Mask mAP50-95 | 0.2336 |
| Box Precision | 0.6701 |
| Box Recall | 0.3972 |
| Box mAP50 | 0.4189 |
| Box mAP50-95 | 0.2568 |

### 相比 V5 的变化

| 指标 | V5 img640 | V6 img768 | 变化 |
|---|---:|---:|---:|
| Mask mAP50-95 | 0.1975 | 0.2336 | +0.0361 |
| Mask mAP50 | 0.3589 | 0.4125 | +0.0536 |
| Mask Precision | 0.5641 | 0.6977 | +0.1336 |
| Mask Recall | 0.3798 | 0.4053 | +0.0255 |

### 结果理解

这次提升是明确的。  
提升主要体现在 precision 和 mAP，说明模型预测更可靠，mask 与真实标签的匹配更好。

但是 recall 只小幅提高，说明模型依然会漏掉不少真实目标。

### 从 V6 得到的结论

图片尺寸从 `640` 提高到 `768` 是有效的。  
V6 成为目前最强、最稳定的 baseline。

---

## V6 后的 Error Analysis

在 `imgsz=768` baseline 之后，我们使用 error analysis notebook 分析错误来源。

### 主要发现

1. **Caries 类明显弱于大目标类别。**  
   Crown、Abrasion 这类较大或更明显的类别表现更好，而 Caries 类整体表现很差。

2. **小目标问题非常严重。**  
   大约 78% 的验证集目标面积小于整张图的 1%。Caries 类尤其小。

3. **训练集和验证集差距明显。**  
   模型在训练集上的表现明显好于验证集，说明存在过拟合或泛化能力不足。

4. **降低 confidence 会增加大量误检。**  
   降低 confidence 可以恢复更多预测，但也会产生很多错误预测。

### 得到的假设

当前瓶颈不只是图片尺寸。  
问题更可能由多个因素共同造成：

- Caries 目标极小；
- 类别不平衡；
- false positives 多；
- 类别混淆；
- 泛化能力不足；
- full-image 训练对很小的牙齿病灶不友好。

---

## V7：YOLOv8s-seg，imgsz=896，rare Caries oversampling，减弱 augmentation

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 896 |
| Epochs | 120 |
| Patience | 25 |
| Oversampling | 对 rare Caries 图片启用 |
| Augmentation | 降低强增强，例如较低 mosaic，并关闭 mixup/copy-paste |
| 主要目的 | 改善小 Caries 检测，并避免强增强破坏小目标细节 |

### 为什么修改这些参数

这版是基于 V6 后的 error analysis 设计的。

当时的思路是：

- Caries 区域很小，因此更高图片尺寸可能有帮助；
- rare Caries 类样本很少，因此 oversampling 可能让模型多看到这些类别；
- 强增强可能破坏小病灶局部细节，因此适当减弱增强可能更适合牙科图像。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 按 Mask mAP50-95 的最佳 epoch | 10 |
| Mask Precision | 0.4760 |
| Mask Recall | 0.4180 |
| Mask mAP50 | 0.3826 |
| Mask mAP50-95 | 0.2260 |
| Box Precision | 0.4930 |
| Box Recall | 0.4397 |
| Box mAP50 | 0.4041 |
| Box mAP50-95 | 0.2539 |

### 相比 V6 的变化

| 指标 | V6 img768 | V7 img896 + oversampling | 变化 |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.4760 | -0.2217 |
| Mask Recall | 0.4053 | 0.4180 | +0.0127 |
| Mask mAP50 | 0.4125 | 0.3826 | -0.0299 |
| Mask mAP50-95 | 0.2336 | 0.2260 | -0.0076 |

### 结果理解

这版让 recall 小幅提高，但是 precision 明显下降。  
说明模型更愿意预测目标，但很多新增预测很可能是 false positives。

因此，提高尺寸、oversampling、减弱增强这个组合没有提高最终分割指标。

### 从 V7 得到的结论

不能直接说明 `imgsz=896` 一定不好，因为这一版同时改了多个变量。  
但是可以说明，这个组合不如 V6 baseline。

所以下一步应该拆分变量，单独测试 `imgsz=896`。

---

## V8：YOLOv8s-seg，单独 imgsz=896

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 896 |
| Oversampling | 关闭 |
| 手动 augmentation 修改 | 移除 |
| Epochs | 120 |
| Patience | 25 |
| 主要目的 | 单独测试从 768 提高到 896 是否有用 |

### 为什么修改这个参数

因为 V7 同时改动了太多变量，所以不清楚变差到底来自：

- `imgsz=896`；
- oversampling；
- 减弱 augmentation；
- 或者这些因素之间的相互影响。

V8 是一个控制变量实验。  
目标是只测试图片尺寸的影响。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch | 10 |
| Mask Precision | 0.4760 |
| Mask Recall | 0.4180 |
| Mask mAP50 | 0.3826 |
| Mask mAP50-95 | 0.2260 |
| Box Precision | 0.4930 |
| Box Recall | 0.4397 |
| Box mAP50 | 0.4041 |
| Box mAP50-95 | 0.2539 |

### 相比 V6 的变化

| 指标 | V6 img768 | V8 img896 only | 变化 |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.4760 | -0.2217 |
| Mask Recall | 0.4053 | 0.4180 | +0.0127 |
| Mask mAP50 | 0.4125 | 0.3826 | -0.0299 |
| Mask mAP50-95 | 0.2336 | 0.2260 | -0.0076 |

### 结果理解

控制变量实验说明：单独把 `imgsz` 从 `768` 提高到 `896` 没有带来提升。

结果模式与 V7 类似：

- recall 小幅提高；
- precision 明显下降；
- Mask mAP50-95 下降。

### 从 V8 得到的结论

`640` 到 `768` 的收益没有继续延伸到 `896`。  
在当前 full-image YOLOv8s 训练方式下，`imgsz=768` 比 `imgsz=896` 更合适。

因此当前阶段不建议继续尝试 `imgsz=1024`。

---

## V9：YOLOv8m-seg，imgsz=768

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8m-seg.pt` |
| 图片尺寸 | 768 |
| Epochs | 120 |
| Patience | 25 |
| 主要目的 | 测试更大模型容量是否能改善表现 |

### 为什么修改这个参数

V8 说明继续提高图片尺寸到 `896` 没有帮助，因此新的假设是：  
`YOLOv8s-seg` 的模型容量可能不足以学习细微的 Caries 特征。

为了测试这个假设，我们把图片尺寸保持在目前最好的 `768`，只把模型从 YOLOv8s-seg 换成 YOLOv8m-seg。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch | 32 |
| Mask Precision | 0.4059 |
| Mask Recall | 0.4050 |
| Mask mAP50 | 0.3889 |
| Mask mAP50-95 | 0.2320 |
| Box Precision | 0.5597 |
| Box Recall | 0.3838 |
| Box mAP50 | 0.3986 |
| Box mAP50-95 | 0.2537 |

### 相比 V6 的变化

| 指标 | V6 YOLOv8s img768 | V9 YOLOv8m img768 | 变化 |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.4059 | -0.2918 |
| Mask Recall | 0.4053 | 0.4050 | -0.0003 |
| Mask mAP50 | 0.4125 | 0.3889 | -0.0236 |
| Mask mAP50-95 | 0.2336 | 0.2320 | -0.0016 |
| Box mAP50-95 | 0.2568 | 0.2537 | -0.0031 |

### 结果理解

YOLOv8m-seg 没有明显提升验证结果。  
Mask mAP50-95 与 V6 几乎持平，但 precision 大幅下降。

这说明单纯增加模型容量不能解决主要瓶颈。  
更大的模型也可能更容易过拟合，或者在这个数据集上产生更不稳定的预测。

### 从 V9 得到的结论

目前最好的 baseline 仍然是：

```text
YOLOv8s-seg + imgsz=768
```

在当前 full-image 训练设置下，更大模型没有明显优势。

---

## V10：YOLOv8s-seg，imgsz=768，轻度 rare Caries oversampling

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 768 |
| Batch size | 16 |
| Epochs | 120（在 epoch 49 时因 patience 提前停止） |
| Patience | 25 |
| Oversampling | 轻度——含 rare Caries 类的图片额外重复 1 次 |
| Augmentation | 默认/baseline 设置 |
| 主要目的 | 测试轻度 rare Caries oversampling 能否在不损害 precision 的前提下改善 Caries 检测 |

### 为什么这样设置

在 full-image YOLO 路线下，所有常规方向都已经测试过：

- 提高图片尺寸到 `896` 没有帮助（V7、V8）；
- 换用更大模型 YOLOv8m 没有帮助（V9）。

唯一还没有单独测试的假设是**轻度** rare Caries oversampling，同时保持目前最好的模型和图片尺寸。  
V7 的强 oversampling 提升了 recall，但 precision 明显下降。  
V10 设计为一个更轻度、更可控的版本：只把含 rare Caries 类的图片额外重复 1 次。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch | 24 |
| Mask Precision | 0.5074 |
| Mask Recall | 0.4685 |
| Mask mAP50 | 0.4089 |
| Mask mAP50-95 | 0.2341 |
| Box Precision | 0.5150 |
| Box Recall | 0.4737 |
| Box mAP50 | 0.4260 |
| Box mAP50-95 | 0.2569 |

训练在 epoch 49 停止，因为 patience=25 在最佳 epoch 24 之后没有新的改善。

### 相比 V6 的变化

| 指标 | V6 img768 | V10 img768 + 轻度 oversampling | 变化 |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.5074 | -0.1903 |
| Mask Recall | 0.4053 | 0.4685 | +0.0632 |
| Mask mAP50 | 0.4125 | 0.4089 | -0.0036 |
| Mask mAP50-95 | 0.2336 | 0.2341 | +0.0005 |

### 结果理解

轻度 oversampling 产生了与 V7 强 oversampling 相同的模式：

- recall 提升；
- precision 明显下降；
- Mask mAP50-95 基本没有变化。

Mask mAP50-95 的 +0.0005 提升在噪声范围内，不能认为是真实的改善。  
模型再次用 precision 换取了 recall：预测了更多目标，但很多新增预测很可能是 false positives。

最佳 epoch 在较早的 epoch 24 出现，训练在 epoch 49 停止且没有进一步改善。  
这说明模型很快达到了性能上限，与此前所有版本观察到的平台期一致。

### 从 V10 得到的结论

轻度 oversampling 没有明显提升 V6 之外的 Mask mAP50-95。  
轻度（V10）和强（V7）oversampling 都呈现相同的权衡模式：recall 上升，precision 下降，严格 mask 指标基本不变。

这证明了 **oversampling 本身无法突破当前的性能上限**。  
full-image YOLO 路线似乎停滞在约 0.23–0.24 Mask mAP50-95。

下一步不应再做 oversampling 或模型大小实验，而是需要对**训练策略进行根本性的改变**。

---

## V11：YOLOv8s-seg，imgsz=768，方案 D（解耦增强 + copy-paste）

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 768 |
| Batch size | 16 |
| Epochs | 60（模板默认；CSV 记录到 epoch 51） |
| Patience | 25 |
| Mosaic | `mosaic=0.0`（关闭） |
| Mixup | `mixup=0.0`（关闭） |
| Copy-paste | `copy_paste=0.2`（启用） |
| 主要目的 | 方案 D——不再让 mosaic/mixup 破坏极小病灶，改用带 mask 的 copy-paste 合成稀有 Caries |

### 为什么这样设置

此前所有方向（图片尺寸、模型大小、oversampling）都停滞在约 0.23–0.24。  
方案 D 的假设是：**破坏性增强**正在伤害我们最关心的极小病灶（mosaic 会把目标缩小，
mixup 会模糊精细的 mask 边界），而 **copy-paste** 是一种"医疗级"增强，把真实病灶
连同其 mask 粘贴到新图中，既能增加稀有 Caries 的曝光，又不会扭曲病灶本身。  
因此计划把两者解耦：关闭破坏性增强，打开 copy-paste。  
通过 `experiments/train_small_object_friendly.py` 实现。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch（按 Mask mAP50-95） | 42 |
| Mask Precision | 0.5656 |
| Mask Recall | 0.4206 |
| Mask mAP50 | 0.3880 |
| Mask mAP50-95 | 0.2135 |
| Box Precision | 0.5747 |
| Box Recall | 0.4288 |
| Box mAP50 | 0.4075 |
| Box mAP50-95 | 0.2372 |

> 注意：这次 run 在 epoch 51 停止（最佳在 42），而 patience=25 要到 epoch 67 才会触发，
> 也就是说 run 被提前中断了。但峰值之后趋势已经在恶化，继续训练也无法补回差距。

### 相比 V6 / V10 的变化

| 指标 | V6 img768 | V10 + 轻度 oversampling | V11 方案 D | V11 vs 最佳 |
|---|---:|---:|---:|---:|
| Mask Precision | 0.6977 | 0.5074 | 0.5656 | — |
| Mask Recall | 0.4053 | 0.4685 | 0.4206 | −0.048 vs V10 |
| Mask mAP50 | 0.4125 | 0.4089 | 0.3880 | −0.021 |
| Mask mAP50-95 | 0.2336 | 0.2341 | 0.2135 | **−0.0206** |
| Box mAP50-95 | 0.2568 | 0.2569 | 0.2372 | −0.020 |

### 结果理解

这是一次**明显的回退**，不是噪声（−0.020 的 Mask mAP50-95 约为我们 ~0.003 噪声带的 7 倍）。  
与 oversampling 实验不同，这次不是 precision/recall 的取舍：Mask mAP50 和 Mask mAP50-95
**同时**下降，说明整体 mask 质量变差了。

训练曲线解释了原因。`train/seg_loss` 全程平滑下降，但 `val/seg_loss` 在第 17 epoch 左右
触底（≈2.09），随后一路上升到 epoch 51 的 ≈2.44——典型的过拟合特征，且比之前的版本更严重。

最可能的原因是**完全关闭了 mosaic**。在 YOLO 里，mosaic 不只是缩小目标的手段，它是最强的
正则化和场景多样性来源。关掉它让这个小数据集更快过拟合，而 `copy_paste=0.2`（少量合成样本）
补不回来。换句话说，mosaic 的正则化价值大于它对小目标缩小的代价——与方案 D 的假设正好相反。

### 从 V11 得到的结论

按此配置，方案 D **损害**了严格 mask 指标。教训不是"copy-paste 不好"，而是
**变量没有解耦**：同时关闭 mosaic 和打开 copy-paste 会混淆结果，而其中 mosaic 的移除起了主导作用。  
应该在**保留 mosaic** 的前提下重新测试 copy-paste，再对 copy-paste 本身下结论。

---

## V12：YOLOv8s-seg + P2 小目标头，imgsz=768

### 训练配置

| 项目 | 设置 |
|---|---|
| 架构 | `yolov8s-seg` + P2 头（4 个分割层，stride 4/8/16/32） |
| 权重 | `.load("yolov8s-seg.pt")` —— backbone 迁移，P2 分支随机初始化 |
| 图片尺寸 | 768 |
| Batch size | 16 |
| Epochs | 120（CSV 记录到 57 epoch） |
| Patience | 25 |
| 数据增强 | 干净的 V6 baseline：`mosaic=1.0`、`close_mosaic=10`、`mixup=0`、`copy_paste=0` |
| Oversampling | 关闭 |
| 主要目的 | 从**架构**层面攻击小目标瓶颈——stride-4（P2）头在 imgsz=768 时给出 192×192 网格（P3 只有 96×96），让最小的病灶落到不止一个 anchor 格子里 |

### 为什么这样设置

V11 证明了增强调整无法突破平台，而图片尺寸、模型大小、oversampling 都已经穷尽。  
V6 后的 error analysis 显示约 78% 的目标面积小于整图 1%，因此 V12 在架构层面直接回应这个发现：
加一个高分辨率 P2 头。增强回退到干净的 V6 baseline，使得 **P2 头是唯一的改动**（单变量原则）。

### 最佳验证结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch（按 Mask mAP50-95） | 32 |
| Mask Precision | 0.5147 |
| Mask Recall | 0.3934 |
| Mask mAP50 | 0.3939 |
| Mask mAP50-95 | 0.2215 |
| Box Precision | 0.5170 |
| Box Recall | 0.3909 |
| Box mAP50 | 0.4085 |
| Box mAP50-95 | 0.2510 |

> **重要警告——最佳值是单轮尖峰。** 在 ep32 所有指标一起跳升
> （Mask mAP50-95 在 ep31/32/33 是 0.1965 → **0.2215** → 0.1946；Box mAP50-95 也跳到 0.251），
> 下一轮立刻掉回去。这是小验证集上的运气 checkpoint，不是稳定水平。最后几轮（ep50–57）
> Mask mAP50-95 维持在 ~0.20–0.212，再也没回到 ep32 的峰值。V12 真实水平诚实地看约为 **0.21**。

### 相比 V6 / V10 的变化

| 指标 | V6 img768 | V10 + 轻度 oversampling | V12 P2 头 | V12 vs 最佳 |
|---|---:|---:|---:|---:|
| Mask Precision | 0.6977 | 0.5074 | 0.5147 | — |
| Mask Recall | 0.4053 | 0.4685 | 0.3934 | **−0.075 vs V10** |
| Mask mAP50 | 0.4125 | 0.4089 | 0.3939 | −0.015 |
| Mask mAP50-95 | 0.2336 | 0.2341 | 0.2215 | **−0.0126**（尖峰）；稳定约 −0.02 |
| Box mAP50-95 | 0.2568 | 0.2569 | 0.2510 | −0.006 |

### 结果理解

V12 **没有**突破平台。即便按 ep32 尖峰取值也比 baseline 低 ≈−0.013，而稳定水平（~0.21）
低 ≈−0.02——与 V11 相当。

最有信息量的信号是 **recall**。P2 头存在的全部理由就是找回极小病灶（~78% 目标 <1% 面积）；
如果它有效，recall 应该上升。但 Mask recall 反而*下降*到 0.393——远低于 V10 的 0.468，
也低于 V6 的 0.405——而且 Mask mAP50 也掉了（0.394 vs 0.41+）。所以这个高分辨率头并没有
多检出小目标；它主要是增加了参数和训练难度，却没带来预期收益。

这不是训练不足造成的假象。P2 分支因为随机初始化，起始 loss 高得多（ep1 `seg_loss` ≈4.68，
而标准头约 2.6），收敛更慢；但它在 ep32 已经追平，之后 25 轮也没再创新高。
`val/seg_loss` 在 ep26 左右触底（~2.26），随后在 2.30–2.45 区间漂移——存在过拟合，但比 V11
那种单调爬升要轻。

### 从 V12 得到的结论

在 **full-image** YOLOv8s-seg 上加 P2 小目标头，无法突破这个数据集 ~0.23–0.24 的平台，
而且明显没能提升 recall——也就是它本该改善的那个指标。结合图片尺寸、模型大小、oversampling
和增强的结果,这强烈说明瓶颈无法靠调整 full-image 模型来解决。下一步必须改变**训练输入**
（crop / tile 裁剪训练），而不是改检测头。

---

## V13：YOLOv8s-seg，crop / tile 裁剪训练（已训练，失败——严重退步）

### 训练配置

| 项目 | 设置 |
|---|---|
| 模型 | 原版 `yolov8s-seg.pt`（撤掉 V12 的 P2 头） |
| 训练输入 | **tile** —— 每张全景图切成重叠的 tile |
| tile 尺寸 | 640 px（同时是训练 `imgsz`，所以 tile 基本 1:1 喂入） |
| 重叠率 | 0.20 |
| 空块保留比例（训练） | 0.15（验证集全保留） |
| 最小面积比例 | 0.35（裁剪后某目标面积不足 35% 就丢弃） |
| 数据增强 | 干净的 V6 baseline：`mosaic=1.0`、`close_mosaic=10`、`mixup=0`、`copy_paste=0` |
| Oversampling | 关闭 |
| Epochs / patience | 120 / 25 |
| 主要目的 | 从**输入**层面攻击小目标瓶颈——这正是 V12 P2 头在架构层面没能解决的问题 |

### 为什么这样设置

V12 证明瓶颈在 **full-image 输入**,而不是检测头:全景图缩放到 768 后,极小病灶只剩几个像素,
所以即便加了高分辨率 P2 头也没有信号可用(recall 没提升)。V13 改的是输入而不是网络:在 tile 上
训练后,一个在全图里只有 ~5 px 的病灶在 tile 里变成 ~20–40 px,模型才有足够像素去学细粒度 mask。
切块输入是相对 V6 baseline 的唯一变量。

### 实现(这就是这次改的代码)

- **`tools/tile_yolo_seg.py`** —— 新增的切块权威库,唯一真相来源:
  - *正向* `build_tiled_dataset`:切图,用 Sutherland-Hodgman 把每个多边形裁剪到 tile 并重新归一化,
    对空白训练 tile 下采样(验证集全保留),写出新的 YOLO-seg 数据集 + `yolo_seg_tiles.yaml`;
  - *反向* `untile_polygon`(tile 归一化 → 全图归一化)和 `merge_detections`
    (按类别做 bbox-IoU NMS,去掉重叠 tile 里的重复目标)。
  - 全图 → tile → 全图的坐标往返已**单元测试**(零误差)。
- **`src/01`** 现在自包含:从头跑到尾会**在 `/kaggle/working` 当场切块并训练**(无需另传 Kaggle Dataset)。
  撤掉 P2 头(原版 `yolov8s-seg`),在 `imgsz=tile 尺寸` 训练,`RUN_NAME` 标 `v13_tile`。切块代码内联镜像,
  保证 Kaggle 自包含。
- **`src/02`** 重写为切块推理:用相同几何切测试图,逐块预测,多边形映射回全图坐标,合并重叠检测,
  写 `submission.csv`。提交格式不变(`id,patient_id,class_id,confidence,poly`,全图归一化多边形),
  且 `02` 的 tile 尺寸/重叠率必须和 `01` 一致。

### ⚠️ 训练时报的 val mAP 不能和 0.234 基线比

训练在**切过块的**验证集上验证,这是个更简单的任务(目标相对更大),所以 `results.csv` 里的 Mask mAP
会偏高,只能用于 early-stopping。和 ~0.234 同口径的成绩必须来自**对完整验证图做切块+合并推理**
(留给后续单独的错误分析 notebook)。

### 结果——严重退步，全项目最差

训练在 epoch 61/120 因 Kaggle 端问题中断，但在切块验证集指标上已经收敛（tiled-val Mask
mAP50-95 最佳约 0.217 @ ep44，ep34 起走平、`val/seg_loss` 开始回升 = 过拟合入口），所以没有续训。
`results/version13_results.csv` 保存了这 61 个 epoch 的曲线。

真正有意义的是**同口径**的数字：用 `best.pt` 对**完整验证图做切块+合并推理**
（notebook `src/03-alphadent-val-map-eval.ipynb`），并且关键在于——V6 baseline 的 `best.pt`
在同样的图上用**同一套**自写 mask-mAP 代码（mask-IoU 匹配 → 10 档 IoU → 101 点 AP）重新评了一遍。
这样 V13 与 V6 的 delta 才是真信号，而不是度量实现差异：

| 指标（同代码，全图） | V13（切块） | V6（原生，重评） | Delta |
|---|---:|---:|---:|
| Mask mAP50    | 0.2428 | 0.3687 | −0.1259 |
| Mask mAP50-95 | **0.0993** | **0.2099** | **−0.1106** |

（V6 重评 = 0.2099，对比历史 0.234，~0.024 的差距是度量实现差异——全分辨率栅格化 + 101 点 AP
vs Ultralytics 内部 val；它对两个模型一视同仁，所以 −0.11 的 delta 远超任何噪声。）

**逐类 Mask mAP50-95——失败完全集中在大目标：**

| 类别 | n_gt | V13 | V6 | Delta |
|---|---:|---:|---:|---:|
| Abrasion | 408 | 0.2342 | **0.6471** | **−0.4129** |
| Crown | 19 | 0.1995 | **0.6313** | **−0.4318** |
| Filling | 186 | 0.1810 | 0.2799 | −0.0989 |
| Caries 1 | 62 | 0.0869 | 0.1195 | −0.0326 |
| Caries 2 | 73 | 0.0320 | 0.0845 | −0.0525 |
| Caries 3 | 33 | 0.0213 | 0.0116 | +0.0097 |
| Caries 4 | 4 | 0.0024 | 0.0000 | +0.0024 |
| Caries 5 | 81 | 0.1290 | 0.1097 | +0.0193 |
| Caries 6 | 5 | 0.0076 | 0.0051 | +0.0025 |

### 失败原因——切块摧毁了撑起分数的大目标

崩盘集中在**大**类（Abrasion −0.41、Crown −0.43、Filling −0.10），由三重机制造成：

1. **训练时大目标被丢弃。** `MIN_AREA_FRAC=0.35` 会把被 tile 边界裁掉、剩余面积 <35% 的实例
   从该 tile 标签里删掉。大结构（Crown、Abrasion）几乎总是跨越 tile 边界，所以它们的大多数实例
   从未进入训练——模型基本没学过。
2. **推理时被切成碎片。** 每个 640 px 的 tile 只看到大目标的一块，预测出的是碎片 mask。
3. **merge 不会拼回去。** `merge_detections` 只去重**重叠**检测（按类 bbox-IoU NMS）；相邻 tile
   里同一个 Crown 的两块碎片几乎不重叠，永远不会被拼回完整目标。碎片 mask 和完整 GT mask 的
   IoU 很低（达不到 0.5 阈值），于是大目标几乎全部漏检。

与此同时，本想受益的极小 Caries 只动了 ±0.01–0.02，而有所改善的类（Caries 3/4/5/6）样本极少
（n_gt 为 4、33、5、81）。小目标的收益从未兑现，即便兑现，这些类权重也太低，无法抵消大目标的崩盘。

### 关键认知修正：mAP 权重 ≠ 目标数量

项目一直奉为圭臬的「~78% 的目标 <1% 面积」描述的是**目标数量**分布，不是 **mAP 权重**分布。
mAP 是**按类平均**的，分数由大类/常见类撑起（V6：Abrasion 0.65、Crown 0.63），而不是稀有、极小的
Caries（它们对**两个**模型 AP 都低，样本还是个位数）。切块恰好牺牲了产生分数的那些类。这也重新
解释了 V6 的平台期和 V12 P2 头的失败：~0.23 主要是大类接近饱和；极小 Caries 既难又权重低，
改进它们几乎不动整体。

### 结论

**V13 是决定性失败（−0.11），是全项目最差结果。** 朴素切块对这个数据集是错误的全局策略，因为它
把主导指标的大目标拱手让出。**V6（≈0.234）仍是最佳模型；提交应继续用它。** 如果仍要攻小病灶，
方案**绝不能牺牲大目标**——例如混合方案：全图模型负责大类，切块只作为小目标的辅助分支，而不是
整体替换全图训练。

---

## 3. 跨版本结论

## 3.1 图片尺寸结论

不同图片尺寸的结果是：

```text
imgsz=640  ->  Mask mAP50-95 = 0.1975
imgsz=768  ->  Mask mAP50-95 = 0.2336
imgsz=896  ->  Mask mAP50-95 = 0.2260
```

结论：

- 从 `640` 提高到 `768` 有帮助；
- 从 `768` 提高到 `896` 没有帮助；
- 目前测试过的尺寸中，`imgsz=768` 最合适。

## 3.2 模型大小结论

在 `imgsz=768` 下对比模型大小：

```text
YOLOv8s-seg + imgsz=768  ->  Mask mAP50-95 = 0.2336
YOLOv8m-seg + imgsz=768  ->  Mask mAP50-95 = 0.2320
```

结论：

- YOLOv8m-seg 没有提升严格 mask 指标；
- YOLOv8m-seg 的 precision 明显更低；
- 单纯增大模型不是主要解决方案。

## 3.3 Oversampling 结论

强 oversampling（V7）和轻度 oversampling（V10）现在都已经测试过。

```text
V7 强 oversampling  ->  Mask mAP50-95 = 0.2260  （recall 上升，precision 下降）
V10 轻度 oversampling  ->  Mask mAP50-95 = 0.2341  （recall 上升，precision 下降）
```

结论：

- 两种程度的 oversampling 都产生了同样的 precision-recall 权衡：recall 提升，precision 下降；
- 无论轻度还是强 oversampling 都没有在 V6 baseline 之外有效提升 Mask mAP50-95；
- oversampling 本身无法解决 Caries 检测瓶颈；
- 在改变底层训练流程之前，不建议再做 oversampling 实验。

## 3.4 过拟合结论

多次训练都出现类似模式：

- train loss 继续下降；
- validation loss 停止改善甚至上升；
- `last.pt` 通常差于 `best.pt`。

结论：

- 后续应该始终使用 `best.pt`；
- 单纯增加 epoch 不是有效方向；
- 未来修改应该提升泛化能力，而不是只训练更久。

## 3.5 数据增强结论（V11 之后补充）

V11 测试了"方案 D"：关闭破坏性增强（`mosaic=0`、`mixup=0`），启用带 mask 的 `copy_paste=0.2`。

```text
V6  baseline（mosaic 开）          ->  Mask mAP50-95 = 0.2336
V11 mosaic 关 + copy_paste 0.2    ->  Mask mAP50-95 = 0.2135  （明显回退，过拟合更严重）
```

结论：

- 在这个小数据集上关闭 mosaic 会移除关键的正则化，加剧过拟合；失去的正则化收益大于
  "不缩小小目标"带来的好处。
- 这个实验同时改了两件事（关 mosaic + 开 copy-paste），所以无法单独评判 copy-paste。
- copy-paste 应该在**保留 mosaic**（或用 `close_mosaic` 只部分关闭）的前提下重新测试。

## 3.6 架构结论（V12 之后补充）

V12 给 YOLOv8s-seg 加了一个 stride-4（P2）小目标分割头，增强回退到干净的 V6 baseline。

```text
V6  baseline（P3/P4/P5 头）        ->  Mask mAP50-95 = 0.2336  （recall 0.405）
V12 + P2 头（stride-4）            ->  Mask mAP50-95 = 0.2215* （recall 0.393）  *单轮尖峰；稳定约 0.21
```

结论：

- 在 full-image 训练下，P2 头**无法**突破 ~0.23–0.24 的平台。
- 最能说明问题的是 **recall 没有提升**（反而下降，0.405/0.468 → 0.393），说明这个额外的
  高分辨率头没能完成它唯一的任务：多检出极小病灶。
- 平台期是**全图输入**的属性，而不是检测头的属性。换头不够，必须改变输入尺度。

## 3.7 输入尺度结论（V13 之后补充）

V13 终于改了输入尺度（切块训练）。在同口径全图指标上（两个模型同一套代码）：

```text
V6  baseline（全图）   ->  Mask mAP50-95 = 0.2099（重评）
V13 切块训练           ->  Mask mAP50-95 = 0.0993（−0.1106）
```

结论：

- 用朴素切块改变输入尺度**不是**答案——它是迄今最差的结果。
- 损失集中在**大**类（Abrasion −0.41、Crown −0.43）：切块在训练时把它们裁出去（`MIN_AREA_FRAC`）、
  推理时切碎、merge 又拼不回来。
- **mAP 权重 ≠ 目标数量**：分数由大类/常见类撑起，所以「小目标瓶颈」的说法高估了上升空间——
  改进极小 Caries 无法抵消丢掉大目标的代价。任何攻小目标的尝试都必须保住大类。

---

## 4. 下一次推荐方向

full-image YOLO 路线下所有常规改进方向都已经测试完毕：

- 图片尺寸：`640` → `768` 有效；`768` → `896` 无效；
- 模型大小：YOLOv8m 没有超过 YOLOv8s；
- Oversampling：轻度（V10）和强（V7）都没有提升 Mask mAP50-95；
- 数据增强（V11）：关闭 mosaic + 加 copy-paste 反而回退（−0.020）；
- 架构（V12）：P2 小目标头没能超过 baseline，且没有提升 recall；
- 输入尺度（V13）：切块训练严重回退（−0.11），因为它摧毁了大目标。

full-image YOLO 路线已出现明显平台期，V12 证明瓶颈不在检测头，V13 又证明朴素切块输入会
**严重回退**（见下）。

### V13 —— crop / tile 裁剪训练（已训练，失败 −0.11）

图片尺寸、模型大小、oversampling、增强和检测头都已穷尽，剩下能动的杠杆就是**输入尺度**。
V13 在重叠 tile 上训练，本想让极小 Caries 病灶在输入中占比大幅提高。结果是**全项目最差**：
同口径 Mask mAP50-95 = **0.0993，对比 V6 重评的 0.2099（−0.11）**，因为朴素切块把贡献了大部分
按类平均 mAP 的大目标（Abrasion、Crown、Filling）切碎并丢弃。完整分析见上面的 V13 小节。

V13 带来的认知修正：**mAP 权重 ≠ 目标数量。**「~78% 的目标极小」说的是目标**数量**；分数由
大类/常见类主导，稀有的极小 Caries 既难又权重太低，撬不动整体。所以驱动 V12、V13 的「小目标
瓶颈」框架本身就指错了方向——改进极小类本就难以提升 mAP，而牺牲大类则是灾难性的。

如果仍要攻小病灶，方案必须**不牺牲大目标**（例如混合方案：全图模型负责大类 + 切块仅作小目标
辅助分支）。一个不触碰大类、工作量更小的备选实验：

- **干净的 copy-paste 消融** —— 保留 `mosaic=1.0`、`mixup=0`、加上 `copy_paste=0.2–0.3`。
  在没有"关闭 mosaic"副作用的情况下单独评估 copy-paste（V11 正是栽在这个副作用上）。

### 当初为什么看好「裁剪训练」（以及为什么适得其反）

V6 后的 error analysis 发现，约 78% 的验证集目标面积小于整张图的 1%。  
在全景图上训练意味着 Caries 病灶在输入中极小，所以切块/裁剪训练当初看起来很有前景。

V13 暴露了缺陷：切块确实让极小病灶在输入中变大，但它**同时破坏了大目标**（被 `MIN_AREA_FRAC`
裁出训练、推理时切碎、merge 又拼不回）。由于大类撑起 mAP，净效果是大幅回退而非提升。

### 推荐配置（V13 之后）

朴素 crop/tile 训练不再推荐——V13 已证明它会摧毁大类。当前选项，按推荐顺序：

1. **保留 V6（≈0.234）作为生产 baseline。** 它仍是最佳模型，提交应继续用它。
2. **混合方案（如仍要攻小病灶）。** 用全图 V6 模型负责大类，切块**仅**作为极小 Caries 的辅助
   分支，再合并——而不是整体替换全图训练。这是唯一不牺牲大目标的切块变体。
3. **干净的 copy-paste 消融（低成本，不触碰大类）。** 保留 `mosaic=1.0`、`mixup=0`，加
   `copy_paste=0.2–0.3`；在没有「关闭 mosaic」副作用的情况下单独评估 copy-paste（V11 正栽在此）。
4. **接受上限。** 鉴于大类已接近饱和、极小 Caries 权重太低，~0.23–0.24 可能就是这个
   模型/数据集的实际上限。

---

## 5. 更长期的优化方向

如果轻度 oversampling 仍然不能提高结果，那么下一步应该改变任务形式，而不是继续简单堆 YOLO 尺寸。

可考虑：

1. **Crop / tile-based training** —— 已作为 V13 尝试，失败（−0.11，见上面 V13 小节）。
   - 朴素全局切块会把贡献大部分 mAP 的大目标切碎/丢弃。作为**全局**策略已被否决；只有当作"保留大目标
     主路径、仅对小目标做辅助分支"时才可能可行。

2. **两阶段 detect-then-refine 流程** —— 已关闭，NO-GO
   （见 [`docs/small_object_research_notes.md`](small_object_research_notes.md)）。
   - 第一阶段：V6 检测器（调高 recall，conf≈0.05）定位框；
   - 第二阶段：训练好的精修器（U-Net + ImageNet ResNet18）对小框 crop 在原分辨率下重新分类+重新分割；
     大框直接走 V6（V13 的护栏）。
   - **状态（2026-06-18）：失败。** Phase 0 oracle 验证的是**天花板**（用完美 GT 框时小 Caries +0.11..+0.22，
     `src/04`），但真实流水线远达不到。Phase 1a 门槛通过（V6 小 Caries 召回 ≈0.58–0.89 @conf0.05）；
     Phase 1b 迁移弱；**Phase 1c**（`src/06`，在真实 V6 框上重训 + 加背景类、热启动，2026-06-18 训练）
     **所有变体都低于 V6 0.2099** —— `full@0.05`=0.157、`TPonly@0.05`=0.178（完美 FP 拒绝的上限）、
     hybrid≈0.203。oracle 的 Caries 增益在真实框上**全部蒸发**；整条差距来自**第一阶段框的质量**
     （conf≈0.05 下召回与定位的矛盾），不是第二阶段的能力。**该方向关闭；V6 仍是生产模型。**
     下一步：转向全类/容量杠杆 —— TTA、V6+V10 集成、更大 backbone。

3. **按类别进行分析**
   - 每一轮都记录 per-class mAP；
   - 重点看 Caries 3、4、5、6 是否改善。

4. **阈值调优**
   - 在 validation set 上调 confidence threshold；
   - 不要简单全局降低 confidence，因为会增加误检。

5. **K-fold 验证**
   - 当前验证集较小，rare class 指标不稳定；
   - K-fold 可以给出更可靠的性能估计。

---

## 6. 当前最佳 baseline

V10 技术上是得分最高的版本，但相比 V6 的提升可忽略不计（+0.0005）。  
V11（方案 D）没能超过它——回退到了 0.2135。  
V12（P2 头）也没能超过它——最佳 0.2215（单轮尖峰；稳定水平约 0.21）。  
V13（crop/tile 裁剪训练）严重回退——同口径 Mask mAP50-95 = 0.0993，对比 V6 重评的 0.2099（−0.11）。  
在新实验明确超过它之前，目前应把下面这版当作主 baseline：

```text
Model:           yolov8s-seg.pt
Image size:      768
Oversampling:    轻度 rare Caries oversampling
Best checkpoint: weights/best.pt
Best Mask mAP50-95: 约 0.2341（V10）
上一个 baseline:   约 0.2336（V6，实际上基本相同）
```

实际使用时，V6 和 V10 都可以作为 baseline 参考。  
V6 的 precision 更高；V10 的 recall 更高。