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
通过独立的 Plan-D 训练模板（`experiments/train_small_object_friendly.py`，V11 关闭后已在 2026-06-24
清理中删除）实现。

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

- **切块库**（`tools/tile_yolo_seg.py`，V13 关闭后已在 2026-06-24 清理中删除；`src/01`/`src/02` 仍内联保留几何代码）曾是唯一真相来源:
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

### V15：NWD 框损失（2026-06-24 已训练，默认 λ=0.5/C=5.0 —— 未达预期）

两条已关闭的精修线（`src/04`–`src/06` 两阶段、`src/07` MedSAM）撞的是**同一堵墙：V6 的小框太松。**
两者都用 GT 框的 **oracle** 验证了小 Caries 有很大的天花板（+0.11–0.22），但真实流水线都够不到——
因为要召回小 Caries 必须用 conf≈0.05，而那个置信度下框定位很松（recall@IoU0.5 ≪ recall@IoU0.3）。
所以下一个杠杆是**在训练阶段直接把框画准**,而不是事后精修。根因：**IoU/CIoU 对小框不稳定**(几像素偏移
就让 IoU 剧烈波动→梯度紊乱→框收不紧),且 IoU 阈值分配让小 GT 分不到正样本。**NWD(归一化高斯
Wasserstein 距离)**把每个框建模成 2D 高斯,对小偏移平滑。V15 把它混进框回归损失:
`box_loss = λ·(1−CIoU) + (1−λ)·(1−NWD)` —— 大框继续靠 CIoU,小框拿到稳定的 NWD 信号。**相对 V6
严格单变量**(只改损失)。实现于 `src/08-yolo-seg-nwd-training.ipynb`(新建训练 notebook,不动 `src/01`)。
完整设计、旋钮、预注册评估(先行指标 = 用 `src/05` 看小 Caries 的 localization recall@IoU0.5)见
`docs/small_object_box_quality_notes.md`。版本号:V14 = MedSAM 评估表,所以这个训练实验是 **V15**
→ `results/version15_results.csv`。

**结果(已训练,默认 λ=0.5/C=5.0 —— 未达预期):** 最佳 Mask mAP50-95 ≈0.24(单轮尖峰;持续 ~0.228
= 贴 V6 平台,无明显增益)。预注册先行指标(`src/05` recall@IoU0.5,V15 vs V6)在所有有支撑的 Caries
(1/2/3/5,均值 −0.035)和大类上都**回退**——λ=0.5 全局混入 NWD 稀释了大/中框依赖的 CIoU 梯度,小框又
没换来收紧。是"这一档参数失败",不是"NWD 死了"(C-sweep {3,5,8}、size-gated NWD 都没试),但该线
**挂起**。真正撬动分数的是全类 **V6+V10 ensemble + 多视角 TTA**(公榜 0.31753 —— 见 §7)。

---


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

3. **MedSAM 掩膜精修** —— Phase 0 已运行（2026-06-23），零样本整体替换 NO-GO
   （见 [`docs/medsam_refine_research_notes.md`](medsam_refine_research_notes.md)；
   结果在 `results/version14_results.csv`，仅评估的 `src/07-medsam-mask-refine.ipynb`）。
   - 与两阶段线**不同**的杠杆：保留 V6 的框 + 类别 + 置信度，只把**粗糙的 YOLO 掩膜换成 MedSAM
     （框提示）掩膜**，目标是**大类的掩膜 IoU**（mAP 权重高、且 V6 大类框可信），不是小目标定位。
   - **结果（所有变体用同一套匹配器）：** `v6box_medsam@0.05` = **0.182 aggregate / 0.499 大类**，
     对比 `v6_native@0.05` = **0.197 / 0.494** —— 大类增益（+0.005）落在噪声带内，且 aggregate 退步
     （−0.015）→ **go/no-go 不满足**。增益是真的，但**只集中在 Abrasion 一个类**（0.618 → 0.665，
     +0.047）；Crown 退步（−0.035），小 Caries **崩塌**（Caries 1/2：约 0.1 → 0.017 —— 框一松 SAM
     就把整颗牙分出来）。GT 框的 `oracle_medsam` = **0.357 / 0.693**（全项目最高 oracle），连小
     Caries 都救了回来 —— 所以瓶颈是**框的质量，不是 MedSAM 掩膜质量**，撞的是和两阶段线同一堵墙。
     **V6 仍是生产模型。** 开放选项：可选的 decoder-only / LoRA 微调（修域差，修不了框差），或转向
     下面的全类杠杆。

4. **按类别进行分析**
   - 每一轮都记录 per-class mAP；
   - 重点看 Caries 3、4、5、6 是否改善。

5. **阈值调优**
   - 在 validation set 上调 confidence threshold；
   - 不要简单全局降低 confidence，因为会增加误检。

6. **K-fold 验证**
   - 当前验证集较小，rare class 指标不稳定；
   - K-fold 可以给出更可靠的性能估计。

---

## 6. 当前最佳 baseline

**最佳提交：V6+V10 ensemble + 多视角 TTA（hflip+vflip+mscale；公榜 0.31753）——见 §7。** 下面汇总的是它所基于的最佳
*单模型*。

V10 技术上是单模型里得分最高的，但相比 V6 的提升可忽略（+0.0005）。V11（方案 D）回退到 0.2135；
V12（P2 头）最佳 0.2215（单轮尖峰，稳定约 0.21）；V13（裁剪/切片训练）严重回退（同口径 0.0993，对比
V6 重评 0.2099，−0.11）；V15（NWD 默认档）停在平台、其框质量领先指标还回退（见 V15 节）。最佳单模型
baseline：

```text
Model:           yolov8s-seg.pt
Image size:      768
Oversampling:    轻度 rare Caries oversampling
Best checkpoint: weights/best.pt
Best Mask mAP50-95: 约 0.2341（V10）
上一个 baseline:   约 0.2336（V6，实际上基本相同）
```

单模型参考时 V6 / V10 皆可（V6 precision 高，V10 recall 高）——但**提交**请用 §7 的 ensemble。

---

## 7. V6+V10 ensemble + 多视角 TTA —— 排行榜增益（公榜 0.31189 → 0.31753）

在小目标的三条线（two-stage、MedSAM、NWD）都撞上同一面"框质量"墙之后，真正撬动排行榜的是**全类、
零训练**的杠杆。分两步：**(7.1)** hflip-only ensemble（首个增益 0.31189）；**(7.2)** 廉价 TTA 视图后续，
加 vflip + 多尺度（当前生产 0.31753）。

**管线。** 全图推理；V6、V10 各对原图**和其水平翻转**预测（检测镜像回原图）；所有检测汇总后做**类内
NMS**（IoU=0.6）。置信度下限**在 val 上调**（`src/10` 用可比 Mask mAP 扫一组阈值，取"最优 0.003 噪声
带内最高的那个"——竞赛是 mAP，硬砍阈值只会截断 PR 曲线，所以阈值要"低但不噪"）。注意 Ultralytics
`augment=True` 对分割模型**无效**（会 warning 并退回单尺度），所以 TTA 用手动 hflip 实现。

**验证（`src/09`，与 src/03/04/05 同口径 Mask mAP）：**

| 变体 | 可比 Mask mAP50-95 | vs V6 锚点 |
|---|---:|---:|
| V6（锚点） | 0.2053 | — |
| V10 | 0.2029 | −0.0024 |
| Ensemble（无 TTA） | 0.2084 | +0.0031 |
| V6 + TTA | 0.2079 | +0.0026 |
| V10 + TTA | 0.2078 | +0.0026 |
| **Ensemble + TTA** | **0.2134** | **+0.0082** |

TTA 单独、ensemble 单独都只在噪声边缘，**只有两者叠加才越过噪声**，所以需要完整的 4 次前向
`Ensemble+TTA`。大类全涨（Abrasion 0.637→0.656、Filling 0.269→0.284、Crown 0.636→0.648），无类显著回退。

**排行榜（7.1，hflip-only）。** `0.31189` 对比单 V6 `0.27047` → **+0.0414**，首个超过单模型 V6 的提交。
公榜涨幅约为 val 可比指标增量（+0.008）的 5 倍——**本地指标明显低估了真实增益**，应把它当作保守的方向性
信号，而非绝对值。

### 7.2. 多视角 TTA 后续 —— `+vflip+mscale`（公榜 0.31753，当前生产）

在 hflip-only ensemble 已是生产配置的前提下，`src/09` §7b 扫了几个廉价、零训练的旋钮，**每个相对
hflip-only `Ensemble+TTA` 基线（val 0.2134）只改一个变量**：合并 `ENS_NMS_IOU` ∈ {0.50, 0.55, 0.60}、
**+vflip** 视图、**+多尺度**（额外 imgsz 640/896）、**两模型置信度加权**。同口径可比 Mask mAP 评分；
`large` 列 = Abrasion/Crown/Filling 平均 AP。判定：val 比基线 > +0.003 **且** 大类不退，才算赢。

| 后续（vs hflip-only Ensemble+TTA = 0.2134） | val mAP | vs 基线 | large |
|---|---:|---:|---:|
| NMS-IoU 0.50 / 0.55 | 0.2129 / 0.2131 | −0.0006 / −0.0003 | 0.529 |
| + vflip（单独） | 0.2123 | −0.0011 | 0.531 |
| + 多尺度（单独） | 0.2161 | +0.0026 | 0.524 |
| **+ vflip + 多尺度** | **0.2173** | **+0.0038** | **0.530** |
| 加权 wV10·0.8 / wV6·0.8 | 0.2113 / 0.2113 | −0.0021 / −0.0022 | 0.524 / 0.528 |

**只有 `+vflip+mscale` 过线**（+0.0038，大类守住 0.530）。注意交互：多尺度是主力（单独 +0.0026），
vflip *单独* 略负（−0.0011）但组合里补了多样性；NMS-IoU 与加权是噪声或负向。接入 `src/10`
（`ADD_VFLIP=True`、`EXTRA_SCALES=[640,896]` → 视图 = orig + hflip + vflip + 640 + 896 = 每图 10 次前向，
约为 hflip-only 的 2.5 倍）。

**排行榜（7.2）。** `0.31753` 对比 hflip-only `0.31189` → **+0.0056**，小但真实、已确认的增益。val 增量
（+0.0038）再次**低估** LB（+0.0056，约 1.5×）——与 7.1 一致，本地指标偏保守。+0.0038 是 83 张 val 上
「8 选 1」的最大值（选择性噪声），所以是 **LB 确认**让它成为生产，而非那个 val 数字。

**状态。** `+vflip+mscale` 是**生产提交**（公榜 0.31753），**零额外训练**。`src/09` 含 §1–8 基线归因、
§7b 后续扫描、§7.3 三模型检验；`src/10` 是提交构建器（自动识别 V6/V10/V9 + 竞赛 test 集，在 val 上调阈值；
`ADD_VFLIP=False`/`EXTRA_SCALES=[]` 退回 hflip-only，不挂 V9 退回 2 模型）。

### 7.3. 三模型集成（加 V9 = yolov8m）—— 已测试,对聚合无帮助（ensemble 线榨干）

更大的**单模型**早已是死路（V9 = yolov8m-seg @768 → 0.232 ≈ V6,见 §V9）。所以 V9 只作为**架构差异更大的
第 3 个集成成员**来试（零训练）：集成增益是**多样性**机制,不是容量。`src/09` 加了 V9 自动识别 + `Ens3`
变体;`src/10` 的 `ensemble_predict` 遍历所有已加载模型,挂上 V9 即变 3 模型（全 TTA 栈 = 每图 15 次前向）。

| 变体（val,vs 2 模型 Ensemble+TTA 0.2134） | mAP | vs 基线 | large |
|---|---:|---:|---:|
| 2 模型 `+vflip+mscale`（生产） | 0.2173 | +0.0038 | 0.530 |
| 3 模型 `Ens3+TTA`（hflip） | 0.2168 | +0.0033 | **0.540** |
| 3 模型 `Ens3 +vflip+mscale` | 0.2154 | +0.0020 | 0.536 |

**3 模型 vs 2 模型（都 `+vflip+mscale`）= −0.0019 → 落在噪声内,无帮助。** 关键细节:V9 **明显抬高大类**
（large 0.529→0.540,hflip 下 +0.011 —— yolov8m 的容量在框已经够好的地方有用）,但**把小 Caries 拖下去**
（小目标召回更低）,而大类在按类平均 mAP 里只占 3/9,一涨一拖抵消,净值略负。这是**同一个 mAP 权重故事**
（V9 的强项砸在近饱和、低边际的大类上）,说明 **ensemble/多样性杠杆已经榨干** —— 连一个架构不同的 yolov8m
都推不动聚合。**结论:保持 2 模型 `+vflip+mscale`（公榜 0.31753）;不提交 3 模型、不挂 V9。** 否决（不值得）:
按类路由（大类用 3 模型、Caries 用 2 模型）、给 V9 降权（加权在 §7.2 已经 val 上失败）。**真正的下一步在
ensemble/容量轴之外** —— 只针对小类、**绕过框**的语义分割 hybrid（唯一逃离框质量墙的杠杆;先跑 oracle 上界）,
或接受 plateau。
---

## 8. 小类分割 hybrid —— 脱离框轴（V16 与路线 B V17 均失败 → 收线）

在 ensemble/多样性杠杆榨干（§7.3）之后,下一条方向彻底离开了画框范式:对**小类（caries）做逐像素分割**,
**大类仍用 V6**（hybrid,这样承载按类平均 mAP、已近饱和的大类不被打扰）。指标是按子类的**实例** Mask
mAP50-95,用与 src/03/04/09 相同的可比 matcher 打分,所以下面每个数字都是真实 delta,不是指标假象。

### 8.1. V16:无框语义分割 hybrid（`src/11`）—— 已跑 & 失败（no-go）

**配置。** `U-Net(resnet18, imagenet)` 在固定 **512×1024** 上做多类逐像素预测 {背景, caries…};CE +
`BG_WEIGHT=0.2` 对抗极端不平衡;按 val 前景 mIoU 存 checkpoint。实例由**逐像素 argmax → 连通域**抽取,
一个连通域一个实例（多边形 = 最大轮廓,**置信度 = 连通域上的平均类概率**）。大类路由给 V6。结果表为
`results/version16_results.csv`（eval-only 的每类 AP，与 V14 同型,不是逐 epoch 曲线）。

**结果 —— 每个有支撑的小类都比 V6 差。**

| 有支撑小类 | semseg（src/11） | V6 参考 |
|---|---:|---:|
| Caries 1 | 0.055 | 0.120 |
| Caries 2 | 0.020 | 0.085 |
| Caries 3 | 0.005 | 0.012 |
| Caries 5 | 0.047 | 0.110 |
| **均值（headline）** | **0.032** | **0.081** |

Hybrid 聚合（9 类）= **0.1855 vs V6 0.2099（−0.024）** —— 把小类路由到更弱的 semseg 分支反而把聚合拖
*低*;大类（Abrasion/Crown/Filling）因为走 V6 基本不变。按预注册 headline,**明确 no-go**。

**诊断 —— 两个独立缺陷相乘。**（1）512×1024 固定 resize 让微小 caries 没有足够像素（分辨率混淆变量）。
（2）语义图→实例的转换在结构上就弱,与像素质量无关:**连通域把"空间相连"当成"同一物体"**（相邻同类
caries 合成一个实例 → 掉 recall;mask 碎裂成两块 → 多一个假阳实例 → 掉 precision）,**平均类概率不是会
排序的分数**（mAP 按置信度排序,排错/"自信的噪声块"分数直接让 AP 崩,即使像素对了）,而且**多类 argmax
让子类逐像素竞争**（边界像素翻类 → 错类实例）。切片/提分辨率只能修缺陷（1）。这就是为什么这条路连 V6 的
*松框*检测器都打不过:脱离框是必要但不充分 —— 实例化与打分必须**学**出来,不能临时拼。

### 8.2. 路线 B:无框 center+offset 实例分割（`src/12`）—— V17,运行 & 失败(NO-GO)

修缺陷（2）:把两个临时拼凑件 —— 连通域实例 + 平均概率置信度 —— 换成**学习式**机器,同时其余一切与
src/11 完全一致,使 headline delta 成为干净的单变量信号。

**它是什么（Panoptic-DeepLab 式）。** 一个 `U-Net(resnet18)` 输出 **`N_SEM + 3`** 通道:`N_SEM` 语义
logits、**1 center heatmap**、**2 offset-to-center**。解码:对 center heatmap 做 max-pool NMS → 实例
中心（**分数 = 峰值**,即学出来的置信度）;每个前景像素投票 `(x+dx, y+dy)` 分给**最近的中心** → 实例
像素组（这能切开粘连的同类病灶,连通域做不到）;实例类别 = 语义多数投票;多边形 = 最大轮廓。损失 =
加权 CE（语义,`BG_WEIGHT=0.2`）+ **CenterNet penalty-reduced focal**（center）+ **masked L1**（offset）。
目标从 GT 多边形按实例构造（栅格化 → 质心 → 高斯 splat + offset 场）;hflip 先作用在**多边形**上,保证
所有目标一致。

**与 src/11 保持一致（单变量）:** 多类语义头、固定 512×1024、`BG_WEIGHT=0.2`、可比 Mask mAP、LARGE→V6
路由。**唯一改动 = 实例抽取 + 打分。**

**搭建时做的抉择**（已向用户说明）:分组 = center+offset（现有 U-Net 的纯 PyTorch 扩展,分数现成;切不开
时退回 StarDist）;center 损失 = CenterNet focal,不是朴素 MSE（稀疏热力图 → MSE 塌成 0）;checkpoint 仍
按 **val fg-mIoU**（语义代理,为干净对照 —— 它不反映实例质量,若结果在边界上,正解是换 center-detection
AP 代理;用户已确认本次保留 fg-mIoU）。留作后续各自单变量实验:二值 caries-vs-bg + 子类头、提分辨率/切片、
Dice/Focal 语义损失。

**预注册读法。** notebook 的 §8 直接打印 `inst − semseg(src11)` 和 `inst − V6`。
- **Go**:`inst` 在有支撑小类上同时超过 src/11（0.032）和 V6（0.081）超过 ~0.003 → 分组+打分就是瓶颈 →
  精修（二值+子类、提分辨率/切片、center-AP checkpoint、TTA）并搭提交路径。
- **Partial**:超过 src/11 但不到 V6 → 机制有效（诊断对了）但像素信号（分辨率）仍封顶 → 下一个单变量杠杆
  是分辨率/切片。
- **No-go**:与 src/11 持平 → 实例/打分机制不是瓶颈 → 去攻分辨率,或收手,保持 2 模型 ensemble（LB
  0.31753）为生产。

设计文档:`docs/instance_seg_small_hybrid_notes.md`。产物:`instance_seg_hybrid_baseline.csv`（存为
`results/version17_results.csv`）+ `instance_seg_small_baseline.pt`。

**结果(V17,应用 v17 FIX 后的重跑 —— `results/version17_results.csv`)。** 第一次(V1)因 decode/checkpoint
缺陷全 0(checkpoint 指标忽略 center head → 无峰值 → 无实例);v17 FIX(checkpoint 改 `fg-mIoU + center-recall`、
`PEAK_THRESH` 0.30→0.05、center 通道 bias 初始化、解码诊断)让重跑变成**公平测试**:

- **实例机制已被验证有效。** 训练达到 **center-recall 0.82**(`loaded best fg-mIoU 0.0189 + c-rec 0.8178`);
  解码产出 **526 个实例 / 83 张图**,center 热图最大值 mean 0.197 / max 0.415,**每图 11.7 个峰值,仅 1/83 张
  图 0 峰值**。center+offset 分组 + 学习到的峰值分数按设计工作 —— 不再是 V1 的零实例失败。
- **但每个有支撑小类 `inst_small_AP = 0.000`,hybrid 聚合 ≈ 0.171**(< V6 0.2099;大类经 V6 路由保持不变 ——
  Abrasion 0.637 / Filling 0.269 / Crown 0.636)。
- **根因 = 像素/mask 质量,不是机制。** `val fg-mIoU` 全程 ≈ **0.008–0.032**(最佳 0.0189)—— 预测的 caries
  像素与真值仅重叠 ~2%。Mask mAP50-95 要求 IoU ≥ 0.5 才算 TP;526 个实例没一个过线 → 各阈值 AP 全 0。
  512×1024 缩放把小 caries 的像素饿死了(§8.1 的分辨率混杂因素)。

这落在预注册的 **No-go** 分支:*分组+打分机制不是瓶颈;像素信号(分辨率)才是。* 留了一个未追的混杂:FIX 的
checkpoint `combo = fg_miou + c_rec` 数值上被 c-rec 主导(0.5–0.8 vs fg-mIoU 0.01–0.03),因此选中了一个
*低* fg-mIoU 的 epoch(0.0189,而 ep39 有 0.0317);重新加权最多把 fg-mIoU 拉回 ~0.03 ≈ src/11,仍然持平 →
判定不值得再训。**收线。** 这是撞上同一堵小类墙的第 4 条线(两阶段、MedSAM、V16 语义、现在 V17 实例);加上
mAP 权重铁律,即便成功上行也有限。**2 模型 V6+V10 ensemble + 多视角 TTA(LB 0.31753)保持为生产模型。**
