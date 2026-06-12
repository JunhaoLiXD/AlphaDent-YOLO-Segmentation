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

目前最好的实用 baseline（V10 技术上最高，但相比 V6 的提升可忽略不计）：

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

---

## 4. 下一次推荐方向

full-image YOLO 路线下所有常规改进方向都已经测试完毕：

- 图片尺寸：`640` → `768` 有效；`768` → `896` 无效；
- 模型大小：YOLOv8m 没有超过 YOLOv8s；
- Oversampling：轻度（V10）和强（V7）都没有提升 Mask mAP50-95。

full-image YOLO 路线已经出现明显平台期。  
下一步应该进行**根本性的训练策略改变**。

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