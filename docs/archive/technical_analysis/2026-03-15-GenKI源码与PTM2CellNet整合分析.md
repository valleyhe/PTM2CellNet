# GenKI 源代码、算法特点与 PTM2CellNet 整合分析

## 结论

GenKI 的核心不是“直接预测 KO 结果”，而是：

`单细胞表达矩阵 -> 构建基因调控图 -> 用 VGAE 学到每个基因的潜在分布 -> 人工制造 virtual KO 图 -> 比较 WT 与 KO 的潜变量分布差异 -> 对受影响基因排序`

所以它本质上是一个“基于图生成模型的虚拟扰动推断框架”。这和 PTM2CellNet 的“序列/PTM -> 细胞状态预测”范式不同，但两者正好互补：GenKI 强在“网络层解释与扰动传播”，PTM2CellNet 强在“蛋白/PTM层表征与状态预测”。

## 一、GenKI 的代码主线

主要分析了这些入口文件：

- `ref/GenKI-master-src/GenKI-master/GenKI/preprocesing.py:65`
- `ref/GenKI-master-src/GenKI-master/GenKI/dataLoader.py:13`
- `ref/GenKI-master-src/GenKI-master/GenKI/pcNet.py:38`
- `ref/GenKI-master-src/GenKI-master/GenKI/model.py:17`
- `ref/GenKI-master-src/GenKI-master/GenKI/train.py:15`
- `ref/GenKI-master-src/GenKI-master/GenKI/utils.py:61`
- `ref/GenKI-master-src/GenKI-master/notebook/Example.ipynb`

代码执行顺序基本是：

### 1. `build_adata`

把表达矩阵读成 `AnnData`，可选加 gene/cell metadata、log normalize，并把原始归一化表达保存在 `adata.layers["norm"]`，标准化表达放回 `adata.X`。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/preprocesing.py:65`

### 2. `DataLoader/scBase`

检查目标基因、细胞类型子集，并决定是加载已有 GRN 还是重新构建 pcNet。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/dataLoader.py:13`

### 3. `make_pcNet`

对 `adata.layers["norm"]` 做 pc regression network，生成基因-基因邻接矩阵。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/pcNet.py:92`

### 4. `load_data` / `load_kodata`

把数据转成 `torch_geometric.data.Data`：

- 节点是基因
- 节点特征是该基因在所有细胞中的表达向量
- 边是 GRN 中的高权重连接

KO 版本会：

- 把目标基因表达置 0
- 删除所有与目标基因相连的边

对应：`ref/GenKI-master-src/GenKI-master/GenKI/dataLoader.py:135` 和 `ref/GenKI-master-src/GenKI-master/GenKI/dataLoader.py:141`

### 5. `split_data`

用 `RandomLinkSplit` 把图上的边切成 train/val/test，任务本质上是 link prediction。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/preprocesing.py:165`

### 6. `VGAE_trainer.train`

训练一个两层 GCN 编码器的 VGAE，目标是：

- 重构正负边
- 加上 `beta * KL` 正则

对应：`ref/GenKI-master-src/GenKI-master/GenKI/train.py:120`

### 7. `get_latent_vars`

训练后分别把 WT 图和 virtual KO 图送进同一个编码器，拿到每个基因的潜变量均值和方差。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/train.py:188`

### 8. `get_distance`

逐基因计算 WT 与 KO 潜变量分布差异，默认是 KL divergence。这个差异就是 perturbation score。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/utils.py:61`

### 9. `pmt` + `get_generank`

通过细胞重排/bootstrap 生成 null distribution，再做 bagging/filtering，得到稳定的基因排名。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/train.py:216` 和 `ref/GenKI-master-src/GenKI-master/GenKI/utils.py:94`

## 二、算法逻辑拆解

从代码和论文看，GenKI 的关键设计有 4 个。

### 1. 图的定义很特别

不是“细胞为节点”，而是“基因为节点、细胞表达向量为节点特征”。

也就是：

- `X_raw`: `cells x genes`
- `Data.x`: `genes x cells`
- `edge_index`: `2 x num_edges`

这让模型学的是“基因在细胞群体中的关系结构”，不是单细胞分类边界。

### 2. 训练目标不是 KO 监督，而是图重构

`model.py` 里的 `InnerProductDecoder` 用潜变量内积重构边，`recon_loss + beta * kl_loss` 是标准 VGAE 结构。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/model.py:17` 和 `ref/GenKI-master-src/GenKI-master/GenKI/model.py:100`

### 3. KO 推断是后验比较，不是前向输出

模型训练只见过 WT 图。KO 推断时并不重新训练，而是：

- 手工改图
- 前向编码一次
- 比较潜分布变化

这使 GenKI 更像“虚拟干预分析器”，不是普通预测器。

### 4. 输出是基因影响排序，不是细胞状态标签

最终 `dis` 是每个基因的 perturbation score，`get_generank` 再转成排名表。

所以它天然适合解释“哪个下游基因最受影响”，而不是直接回答“细胞会变成什么状态”。

## 三、代码层面的算法特点

### 优点

- 无监督或弱监督。只需要 WT scRNA-seq，不需要真实 KO 配对样本。
- 解释性强。KO 是“清零表达 + 删除相关边”，非常直观。
- 图结构和表达特征一起用，比只看相关性更强。
- 代码里还扩展了 OE 路径，`OE_data_init` 可以模拟 overexpression。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/dataLoader.py:205`

### 限制

- `pcNet` 很重。`pcNet.py` 对每个基因重复做 SVD，计算量大，基因数一上去就贵。
- 泛化依赖 cell context。节点特征维度就是“细胞数”，换个数据集、换个细胞群体，输入语义就变了。
- KO 模拟是硬规则，不是生物化学机制模型。对 PTM 这种“部分调节”不够细。
- 排名分数依赖对角高斯近似，`get_distance` 只用对角协方差。

对应：`ref/GenKI-master-src/GenKI-master/GenKI/pcNet.py:23` 和 `ref/GenKI-master-src/GenKI-master/GenKI/utils.py:67`

## 四、数据流

可以把 GenKI 的数据流写成：

`AnnData(counts)`
`-> build_adata(): 标准化表达 + norm layer`
`-> make_pcNet(): gene-gene network`
`-> DataLoader.load_data(): WT graph`
`-> DataLoader.load_kodata(): virtual KO graph`
`-> split_data(): edge split`
`-> VGAE train: learn gene latent distributions`
`-> get_latent_vars(WT, KO)`
`-> get_distance(): per-gene KL/EMD/t-stat`
`-> pmt(): bootstrap null`
`-> get_generank(): final ranked affected genes`

维度上最关键的是：

- `adata.X`: `n_cells x n_genes`
- `data.x`: `n_genes x n_cells`
- `z_mu, z_var`: `n_genes x latent_dim`
- `dis`: `n_genes`
- `null`: `n_perm x n_genes`

## 五、论文与代码的一致处和差异

### 一致处

- 论文主方法“GRN + VGAE + virtual KO + latent distribution shift”与源码主干一致。
- notebook 示例也严格走这条链：`build_adata -> DataLoader -> VGAE_trainer -> get_distance -> get_generank`。

对应：`ref/GenKI-master-src/GenKI-master/notebook/Example.ipynb`

### 值得注意的实现细节

- 训练时每轮监控的是 `test_data`，不是 `val_data`。
- CLI 参数定义成了 `--ddir`，但运行时读的是 `args.dir`，这个入口按源码看有 bug。
- `make_pcNet` 虽然支持 `ray`，但实际只把整个 `pcNet` 扔成一个远程任务，不是按基因并行拆分。

对应：

- `ref/GenKI-master-src/GenKI-master/GenKI/train.py:150`
- `ref/GenKI-master-src/GenKI-master/GenKI/train.py:257`
- `ref/GenKI-master-src/GenKI-master/GenKI/train.py:311`
- `ref/GenKI-master-src/GenKI-master/GenKI/pcNet.py:92`

## 六、对 PTM2CellNet 的整合判断

先说结论：不能“直接拼接”，因为两边输入空间不同。

- GenKI 输入：单细胞表达矩阵 + target gene + 细胞上下文
- PTM2CellNet 输入：蛋白序列 + PTM 位点 + 细胞状态标签

中间至少需要一个桥接层：

`protein / PTM -> gene/protein activity prior -> GRN perturbation -> cell-state consequence`

PTM2CellNet 当前主干是：

- `src/models/architectures.py:22`
- `src/models/ptm_modules.py:61`
- `src/data/datasets.py:20`

它已经有“序列编码器 + PTM 融合 + 分类头”的干净边界，适合加一个 graph branch 或 auxiliary target。

## 七、3-5 种整合参考思路

### 1. 两阶段筛选：PTM2CellNet 先筛，GenKI 后解释

先用 PTM2CellNet 预测“哪些 PTM/蛋白最可能改变细胞状态”，再把映射到的 gene 输入 GenKI 做 virtual KO，得到下游基因排名和通路解释。

这是最稳妥的整合，因为几乎不改现有主干，只是在 `src/models/architectures.py:121` 之后增加一个后处理解释层。

### 2. 把 GenKI 输出当作 PTM2CellNet 的辅助监督

如果有“某 PTM 改变对应的单细胞表达数据”，可以先跑 GenKI 得到目标基因影响向量或排名，再让 PTM2CellNet 额外预测这个 perturbation signature。

这样 PTM2CellNet 不只学分类标签，还学“网络传播结果”，更适合小样本生物任务。

### 3. 做 PTM-aware virtual perturbation，替代 hard KO

GenKI 现在的虚拟干预是“表达置零 + 删除边”，更像基因缺失。PTM 场景可以改成“边权缩放 + 节点活性衰减”，用 PTM 类型和位点决定扰动强度。

这条路最贴近生物机制，也最适合本项目，因为 PTM2CellNet 现有 `PTMModule` 已经能输出位点级表征，参考 `src/models/ptm_modules.py:61`。

### 4. 给 PTM2CellNet 增加 graph-conditioned cell-context branch

保留现有序列/PTM 编码分支，同时引入一个 GenKI/GRN 分支编码细胞类型特异网络，把两者在 predictor 前融合。

实现上最自然的切入点是 `src/models/architectures.py:147` 之后、`src/models/architectures.py:148` 之前，把 `fused` 和 graph embedding 做 `concat` 或 `cross-attention`。

### 5. 做候选优先级系统而不是端到端单模型

把 PTM2CellNet 当“上游分子事件评分器”，GenKI 当“下游网络效应模拟器”，最终输出：

`PTM event score + network impact score + cell-state confidence`

这比强行把两个模型揉成一个更实用，也更容易做实验验证。

## 八、推荐实施顺序

1. 先做“两阶段筛选/解释”。
2. 再做“GenKI 伪标签辅助监督”。
3. 如果数据足够，再做“PTM-aware virtual perturbation”。

## 参考来源

- 本地源码压缩包：`ref/GenKI-master.zip`
- 本地解压源码：`ref/GenKI-master-src/GenKI-master`
- PTM2CellNet 主架构：`src/models/architectures.py:22`
- PTM2CellNet PTM 融合：`src/models/ptm_modules.py:61`
- PTM2CellNet 数据集接口：`src/data/datasets.py:20`
- GenKI GitHub 仓库：https://github.com/yjgeno/GenKI
- GenKI 示例 notebook：https://github.com/yjgeno/GenKI/blob/master/notebook/Example.ipynb
- GenKI 论文页面：https://academic.oup.com/nar/article/51/13/6578/7184155?login=false

## 说明

本分析以本地压缩包源码为主，辅以仓库 README、示例 notebook 和论文页面信息。由于论文主页对自动抓取有限制，本报告对论文部分主要基于源码实现、仓库说明与公开页面元信息做交叉归纳。
