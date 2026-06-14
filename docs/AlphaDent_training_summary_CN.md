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

\* V12 的 0.2215 是 ep32 的单轮尖峰；稳定水平约 0.21，即相对 V6/V10 baseline ≈−0.02。

目前最好的实用 baseline（V10 技术上最高，但相比 V6 的提升可忽略不计；V11 和 V12 都没能超过它）：

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

---

## 4. 下一次推荐方向

full-image YOLO 路线下所有常规改进方向都已经测试完毕：

- 图片尺寸：`640` → `768` 有效；`768` → `896` 无效；
- 模型大小：YOLOv8m 没有超过 YOLOv8s；
- Oversampling：轻度（V10）和强（V7）都没有提升 Mask mAP50-95；
- 数据增强（V11）：关闭 mosaic + 加 copy-paste 反而回退（−0.020）；
- 架构（V12）：P2 小目标头没能超过 baseline，且没有提升 recall。

full-image YOLO 路线已出现明显平台期，V12 进一步证明瓶颈不在检测头。  
下一步必须**从根本上改变训练输入**。

### 下一次实验（V13 —— crop / tile 裁剪训练）

图片尺寸、模型大小、oversampling、增强和检测头都已穷尽，剩下能动的杠杆就是**输入尺度**。
crop / tile 裁剪训练是对"~78% 目标 <1% 面积"最对口的回应：在牙齿/区域级 crop 上训练，
极小 Caries 病灶在输入中占比大幅提高，模型可以学到更细粒度的 mask——这正是 P2 头想在架构层面
实现、却失败了的目标。

如果裁剪工具还没准备好，可先做一个工作量更小的替代：

- **干净的 copy-paste 消融** —— 保留 `mosaic=1.0`、`mixup=0`、加上 `copy_paste=0.2–0.3`。
  在没有"关闭 mosaic"副作用的情况下单独评估 copy-paste（V11 正是栽在这个副作用上）。

最优先推荐的方向是：

```text
Crop / Tile-based training（局部裁剪训练）
```

### 为什么选择这个方向

V6 后的 error analysis 发现，约 78% 的验证集目标面积小于整张图的 1%。  
在全景图上训练意味着 Caries 病灶在输入中极小，模型很难学习到细粒度的分割特征。

Crop-based training 直接解决这个问题：

- 用牙齿局部区域或可疑区域的 crop 代替全景图来训练；
- 这样 Caries 病灶在输入中占比大幅提高；
- 模型可以学习到更细粒度的分割特征。

### 推荐配置

| 项目 | 建议设置 |
|---|---|
| 模型 | `yolov8s-seg.pt` |
| 图片尺寸 | 640 或 768（相对 crop 尺寸） |
| 训练数据 | 从全景图中裁剪的牙齿局部区域 |
| Epochs | 100–150 |
| Patience | 25–30 |
| Augmentation | 默认/baseline 设置 |

### 如果 crop 训练暂时无法实现

如果 crop-based training 尚未准备好，下一步也可以先做详细的 per-class mAP 分析，了解哪些 Caries 子类是主要瓶颈，再结合 validation set 上的 confidence threshold 调优。

---

## 5. 更长期的优化方向

如果轻度 oversampling 仍然不能提高结果，那么下一步应该改变任务形式，而不是继续简单堆 YOLO 尺寸。

可考虑：

1. **Crop / tile-based training**
   - 不用整张全景图训练，而是训练牙齿局部区域。
   - 这样 Caries 病灶在输入图中占比会更大。

2. **两阶段流程**
   - 第一阶段：检测牙齿或可疑区域；
   - 第二阶段：在局部 crop 内做分割和分类。

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