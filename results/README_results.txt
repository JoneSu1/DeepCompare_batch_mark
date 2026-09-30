DeepCompare_batch_mark — 结果交付说明（2026-09-30 全量运行, Colab T4）
=================================================================

correlations.csv        144 行 = 8 元件 × 18 轨道 的 Pearson 相关系数
                        （Element, Cell, Method, Track, Correlation, N）
figure_correlations.png/.pdf   出版图（2×4 分面, 按任务书参考图版式）
figure_caption.md       图注（中英文）
predictions/deepcompare.csv   DeepCompARE 8 轨道 ref→mut 预测差值 (3877×8)
predictions/enformer.csv      Enformer 13 条所选轨道, 中心 bin 448 (3877×13)
predictions/alphagenome.csv   AlphaGenome 8 条原始轨道 (3877×8)

数据: GRCh38_F9_GP1BA_HBB_HBG1_LDLR_PKLR-24h_BCL11A_SORT1.tsv
过滤: Alt != '-' 且 P-Value < 0.05 → 3,877 个单碱基变异
模型: DeepCompARE (本地 model.h5) / Enformer (HF: EleutherAI/enformer-official-rough)
      / AlphaGenome (DeepMind API)
复现: GitHub 仓库 README → 本地三行命令, 或 notebooks/run_colab.ipynb 一键

重画图: python scripts/06_figure.py (需 pandas/matplotlib, 依赖 src/ 与 configs/ 一并附上)
