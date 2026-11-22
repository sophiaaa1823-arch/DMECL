# DMECL



## 目录

```text
DMECL/
├─ config/main.yaml
├─ data/preprocess_pddca.py
├─ data/pddca.py
├─ data/raw/PDDCA/
├─ data/processed/PDDCA/
├─ models/dmecl.py
├─ losses/dme.py
├─ training/train.py
├─ testing/test.py
├─ testing/metrics.py
├─ inference/infer.py
├─ inference/predict.py
├─ requirements.txt
└─ README.md
```

## 环境

使用 Python 3.10 及以上版本，并安装与显卡驱动匹配的 PyTorch：

```powershell
python -m pip install -r requirements.txt
```

## PDDCA 数据准备

将原始 PDDCA 病例解压至 `data/raw/PDDCA/`，每个病例占一个子目录。病例目录中放 CT 的 `.nrrd` 文件，器官掩膜可位于该目录的子目录内。预处理器按文件名识别九个器官，读取 NRRD 自带的物理空间信息，用线性插值重采样 CT、最近邻插值重采样掩膜，输出保留几何信息的 `.npz` 文件。缺失器官会记录在 `available` 中，不会伪造标注。

类别编号：0 背景、1 下颌骨、2 脑干、3 左腮腺、4 右腮腺、5 左颌下腺、6 右颌下腺、7 左视神经、8 右视神经、9 视交叉。左右类别遵循文件名的 L/R 标记。

```powershell
python data/preprocess_pddca.py --raw-dir data/raw/PDDCA --output-dir data/processed/PDDCA
```

预处理会输出 `manifest.json` 与 `splits.json`。程序使用固定随机种子划分 32 例训练、16 例测试；训练集再分成 2 例有标注、4 例验证、26 例无标注。有标注与验证病例从完整标注病例中选取。

## 训练、测试和推理

```powershell
python training/train.py --config config/main.yaml
python testing/test.py --config config/main.yaml
python inference/infer.py --config config/main.yaml --input data/processed/PDDCA/病例ID.npz --output prediction.nrrd
```

训练保存 `checkpoints/last.pt` 和验证 Dice 最优的 `checkpoints/best.pt`。测试结果保存到 `checkpoints/test_metrics.json`。推理默认输出回到原始 CT 几何空间的 NRRD 标签图；指定 `--format npz` 可输出预处理空间的 NumPy 预测数组。

## 方法与可配置参数

总损失为 `Lsup + αLdme + βLst + γLent`。对比损失实现论文式 (2)–(5)：正样本为学生与教师对应像素，负样本需有不同的十类伪标签；按前景比例小于 0.3 或大于 0.7 的补丁采样，并保留前景、背景两类特征库。正样本角度加入按特征范数变化的边距，随后经论文的缩放函数和温度参数计算对比项与对齐项。

`config/main.yaml` 中，32/16/2/4/26 划分、100+50 个 epoch、每 epoch 8 次迭代、批量 8、Adam 和学习率 0.001/0.0001 

默认重采样间距为 `(0.76, 0.76, 1.25) mm`，取自 [AnatomyNet 的 PDDCA 处理说明](https://github.com/wentaozhu/AnatomyNet-for-anatomical-segmentation)。九类映射和缺失标注记录方式依据 [PDDCA-Lite 数据说明](https://huggingface.co/datasets/YongchengYAO/PDDCA-Lite/blob/main/README.md)。预处理使用 SimpleITK 直接读取、写回 NRRD 几何，不经过 NIfTI 坐标系转换。训练切片统一缩放至配置的 `image_size`；测试和推理会先把结果缩回重采样切片大小，再按保存的原始几何信息恢复输出。

论文报告九个解剖结构，其中表 1 统计六个结构，表 2 统计视神经与视交叉。逐一报告全部可用类别，并输出 `mean`、`table1_mean`、`table2_mean`；缺失标注不参与对应类别的平均值。