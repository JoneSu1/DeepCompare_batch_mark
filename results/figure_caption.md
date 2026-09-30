# Figure 1 caption — MPRA mutation effect size prediction

## English

Figure 1. Benchmark of MPRA mutation effect size prediction. For each of eight
regulatory elements (panel titles, with assayed cell line), the Pearson
correlation is shown between the predicted reference-to-mutant difference and
the measured satMutMPRA effect size (single-nucleotide variants with P < 0.05;
n = 244-1,155 per element). Points represent cell-type-matched prediction
tracks for each model: DeepCompARE (CAGE, DNase, STARR, SuRE; 4 points),
Enformer (mean of all CAGE and all DNase tracks of the element's cell line,
scored at the center 128-bp bin of a +/-98.3-kb context; 2 points), and
AlphaGenome (CAGE mean, DNase mean, and cell-type-specific ATAC, aggregated
over the element-centered 600-bp window; 3 points). Marker shape denotes the
prediction method; colour denotes the model. Black lines mark each model's
median within an element. Across all element-track points, median correlations
are 0.61 (AlphaGenome), 0.47 (Enformer), and 0.45 (DeepCompARE); DeepCompARE
performs strongest on SuRE/STARR functional tracks (up to r = 0.79, PKLR-24h
SuRE), whereas Enformer shows negative CAGE correlations for HBG1 (r = -0.64)
and HBB (CAGE HepG2, r = -0.29).

## 中文

图 1. MPRA 突变效应量预测基准测试。8 个调控元件（面板标题，括号内为对应细胞系）
分别展示模型预测的参考→突变差值与 satMutMPRA 实测效应量的 Pearson 相关系数
（单碱基变异，P < 0.05；每元件 n = 244-1,155）。每个点代表该元件细胞系匹配的
一条预测轨道：DeepCompARE 4 点（CAGE/DNase/STARR/SuRE）；Enformer 2 点（该细胞系
全部 CAGE 轨道均值、全部 DNase 轨道均值，取 +/-98.3 kb 上下文中心 128 bp bin）；
AlphaGenome 3 点（CAGE 均值、DNase 均值、细胞特异 ATAC，在元件中心 600 bp 窗口内
聚合）。点的形状 = 预测方法，颜色 = 模型；黑色短横线 = 该元件内各模型的中位数。
全部元件-轨道点的中位相关系数：AlphaGenome 0.61 > Enformer 0.47 >
DeepCompARE 0.45；DeepCompARE 在 SuRE/STARR 功能测定轨道上最强（PKLR-24h SuRE
达 r = 0.79），Enformer 在 HBG1（CAGE K562 r = -0.64）与 HBB（CAGE HepG2
r = -0.29）出现负相关。
