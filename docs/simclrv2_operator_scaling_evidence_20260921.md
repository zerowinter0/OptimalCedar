# simclrv2 逐算子“单条数据字节量 → 单条处理耗时”曲线数据（2026-09-21）

本文件回答：simclrv2 负载（`evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py` 的
`SimCLRv2Feature`）上，是否测过**每个算子随着单条输入字节量增加，其单条数据处理耗时/速度的变化曲线**。
结论：**测过，共四批可用数据**，口径和用途不同。下面按“原始测量 → 拟合系数 → 进 profile 的 sweep →
论文动机图”的顺序给出路径、数字与使用限制。

## 0. 数据资产总览

| # | 用途 | 位置 | 内容 |
| --- | --- | --- | --- |
| 1 | 受控尺寸 sweep 原始测量（9 算子 × 12 尺寸 × 3 轮 × 3 后端） | `outputs/target_pipeline_scaling_20260910/` | `PROTOCOL.md`、`results/raw.csv`、`results/operators.json`、latency/吞吐图 |
| 2 | simclrv2 仿射拟合对照（k、b、R²） | `outputs/simclrv2_affine_comparison_20260911/` | `README.md`、`coefficients.csv`、`local_measurements.csv`、`comparison.png/pdf` |
| 3 | 写进 profile 的逐算子多尺寸 sweep | `outputs/simclrv2_scaling_20260911/profile_calibrated.yaml` 的 `cm_model` | 每算子 13 个尺寸点 + train/validation + R² |
| 4 | 论文动机图的 4 点算子缩放 formal run | `outputs/motivation_multimodal/simclrv2_multimodal_operator_scaling_formal_20260909/` | 6 算子 × 4 work_scale × 7 trials，`summary.csv`、`raw.json`、`figures/` |

## 1. 原始测量：受控尺寸 sweep（`outputs/target_pipeline_scaling_20260910/`）

协议见 `PROTOCOL.md`，要点：

- 五个 workload 的所有非 source 算子按 batch size 4 抽出；simclrv2 共 9 个算子
  （`op1_RGBReader`、`float`、`crop`、`op4_RandomHorizontalFlip`、`jitter`、`op6_Grayscale`、
  `op7_GaussianBlur`、`op8_Normalize`、`op9_BatchKernel`），每个出现位置独立测量不去重。
- 12 个线性像素尺寸点（64² → 2048²，方图保持长宽比）；字符串 256 → 65536 UTF-8 字节线性 12 档；
  token 表示 16 → 4096 元素线性 12 档。Reader 的 x 轴是**编码文件字节**（未压缩 PNG），
  其余图像算子是**解码后字节**。
- 输入内容是 4 组真实 COCO 图文对，图像在计时外完成 resize：这是**受控反事实输入**，
  不是自然分布。
- 每个 operator/size/resource 各 3 轮，资源顺序轮换（local/smp/ray）；每格 4 条 warmup，
  校准目标 30 ms、调用数 4..8192，且同一格三个后端用相同调用数。
- 计时发生在 Cedar 原生 worker 内（`InProcessBatcherPipeVariant` / SMP / Ray actor），
  排除数据准备、传输、worker 启动与 warmup；测的是**算子计算**，不是端到端 stage 吞吐。
- 产出：`results/raw.csv`（约 1.7 万行，含 `input_bytes`、`ms_per_record`、`mib_per_sec`、
  `calls`、`seconds`、worker `pid`、`torch_threads`）、`results/operators.json`（161 个算子定义）、
  `results/figures/*.pdf`（按 backend/modality 输出 latency 与逻辑 MiB/s 曲线）、
  `results/metadata.json`、`results/index.html`。启动门禁与失败重跑记录见 `gate/` 与 `launcher_resume.log`。

## 2. 拟合对照：9 个算子的 k、b、R²（`outputs/simclrv2_affine_comparison_20260911/`）

该目录用 optimizer 的同一个非负仿射拟合器（`cedar.client.linear_cost_profile.fit_affine`，
12 等宽 bin）重拟合上面的 local 后端数据：9 算子 × 12 尺寸 × 3 轮 = 324 条观测，模型为
`t[ms/record] = k · x[MiB] + b`。产物：

- `local_measurements.csv`：逐 trial 的 `input_bytes / ms_per_record / mib_per_sec`；
- `coefficients.csv`：每个算子 sweep 拟合的 k、b、R²、实测尺寸范围，以及当前 DP/CM profile 系数的对照；
- `wide_range_models.yaml`、`comparison.png`、`comparison.pdf`、`README.md`（含下表与注意事项）。

| 算子 | sweep k (ms/MiB) | sweep k (ms/byte) | sweep b (ms) | R² | 实测范围 (MiB) | 与 profile 可直接比较 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ImageReaderPipe | 5.43166 | 5.18003e-06 | 0.888492 | 0.9992 | 0.0120 – 12.0060 | 否（file bytes vs path 对象字节，PNG vs JPEG） |
| to_float | 1.15334 | 1.09991e-06 | 0 | 0.7145 | 0.0117 – 12.0000 | 是 |
| RandomResizedCrop | 0.163394 | 1.55825e-07 | 0.963732 | 0.7970 | 0.0469 – 48.0000 | 是 |
| RandomHorizontalFlip | 0.252091 | 2.40413e-07 | 0 | 0.8986 | 0.0469 – 48.0000 | 是 |
| ColorJitter | 21.3959 | 2.04047e-05 | 0 | 0.9452 | 0.0469 – 48.0000 | 是 |
| Grayscale | 0.353303 | 3.36936e-07 | 0 | 0.9655 | 0.0469 – 48.0000 | 是 |
| GaussianBlur | 43.9257 | 4.18908e-05 | 0 | 0.6018 | 0.0156 – 16.0000 | 是 |
| Normalize | 0.385662 | 3.67796e-07 | 0 | 0.8765 | 0.0156 – 16.0000 | 是 |
| BatcherPipe(batch_size=4) | 0.46338 | 4.41913e-07 | 0 | 0.8870 | 0.0156 – 16.0000 | 否（batch 号与计时方法不同） |

读法：斜率相差两个数量级（ColorJitter 21.4 ms/MiB、GaussianBlur 43.9 ms/MiB，而 Crop 只有
0.163 ms/MiB）；GaussianBlur（0.60）与 to_float（0.71）的线性拟合质量明显偏低，
引用时必须标注 R²，不能当成精确线性。

## 3. profile 内嵌 sweep：每算子 13 个尺寸点

`outputs/simclrv2_scaling_20260911/profile_calibrated.yaml` 的 `cm_model` 记录了构建该 profile 时的
受控图像 sweep（`input_policy: natural_capture_plus_controlled_sweep`、`input_scope: whole_record`），
每个算子 13 个尺寸点、3 轮、含 train/validation 划分与 timing CV：

| 算子（cm_model） | status | k (ms/byte) | b (ms) | R² | 实测字节范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| MapperPipe_Normalize | fitted | 2.0436e-07 | 0.052187 | 0.9997 | 16,384 – 1,110,916 |
| MapperPipe_GaussianBlur | fitted | 2.7871e-05 | 3.374316 | 0.7485 | 16,384 – 1,110,916 |
| MapperPipe_Grayscale | fitted | 1.8813e-07 | 0.063881 | 0.9988 | 49,152 – 3,332,748 |
| MapperPipe_ColorJitter | fitted | 1.1951e-05 | 0.232923 | 0.9976 | 49,152 – 3,332,748 |
| MapperPipe_RandomHorizontalFlip | fitted | 4.4907e-08 | 0.036545 | 0.9873 | 49,152 – 3,332,748 |
| MapperPipe_RandomResizedCrop | fitted | 1.1133e-07 | 1.557217 | 0.9450 | 48,810 – 8,502,539 |
| MapperPipe_to_float | fitted | 2.1985e-07 | 0.005676 | 0.9991 | 12,203 – 2,125,619 |
| BatcherPipe(batch_size=1) | constant_fallback | 0 | 0.008919 | — | 238,144（无尺寸变化） |
| ImageReaderPipe | constant_fallback | 0 | 2.057682 | — | 152 – 168（path 对象字节不可预测解码耗时） |
| LocalFSListerPipe | unavailable | — | — | — | — |

同一族的 wide-range 版本（batch 4）在 `outputs/simclr_profile_wide_scaling/simclr_profile.yaml`。
两个 run 的原始 `observations` 均保留每点 `input_bytes`、`mean_ns`、`round`、`split`、`target_pixels`，
可直接重画曲线。注意：`outputs/simclrv2_scaling_20260911/status.json` 记录该 campaign 的 driver 步骤
`failed`（profile 产物本身已完整落盘），引用时不要把它说成一次完整跑通的对比实验。

现行协议（2026-09-19 起）不再把 13 点曲线放进供 optimizer 消费的 profile：
`outputs/six_workload_profile_final_20260919/simclrv2/profiles/shared.yaml` 的
`physical_model.operator_affine`（约 2678 行起）对每个算子只保留**最小/最大两个合法输入点**
（`points_ms_per_byte`）加 k、b、`fixed_fraction`；0–7 有拟合，8 因
`unable to mmap 4096 bytes` 测量失败，9 是 `not_a_single_input_stage`。

## 4. 论文动机图的 4 点算子缩放（`outputs/motivation_multimodal/simclrv2_multimodal_operator_scaling_formal_20260909/`）

6 个算子 × 4 个 work_scale（0.25/1/4/16）× 7 次重复 = 24 点、168 行原始数据，输入尺寸对应
文本 16/64/256/1024 词与图像边长 112/224/448/896。`summary.csv` 给出每点
`median_ns_per_record`、`median_records_per_sec` 与四分位；单条耗时（ms/record，中位数）为：

| 算子 | 0.25× | 1× | 4× | 16× |
| --- | ---: | ---: | ---: | ---: |
| normalize | 0.03044 | 0.05991 | 0.1723 | 0.6094 |
| clip | 18.28 | 17.95 | 22.32 | 36.03 |
| random_crop | 1.455 | 1.483 | 1.605 | 2.061 |
| random_flip | 0.02094 | 0.03536 | 0.07263 | 0.2995 |
| color_jitter | 2.402 | 7.608 | 32.56 | 164.4 |
| gaussian_blur | 16.15 | 58.72 | 236.8 | 989.1 |

同目录 `figures/figure1_operator_scaling.{png,pdf,svg}` 与 `figure_manifest.json`（含 sha256）。
单模态维度版本在 `outputs/motivation_multimodal/simclrv2_multimodal_modality_scaling_formal_20260909/summary.csv`，
论文当前引用的就是它：`my_paper/69e75a0100d7b4afeb1cfc20/section/03_cost_model.tex` 的
`figures/operator_modality_scaling.pdf`；由算子版 summary 生成的
`figures/figure1_total_input_scaling.*` 目前正文未引用，溯源见 `figures/figure_manifest.json`。

## 5. 使用限制

- 曲线来自**受控反事实输入**（图像在计时外 resize、文本按真实 caption 重复构造），不是自然分布，
  不能当作“真实数据里就会遇到这些尺寸”的证据；引用时应说明是受控 sweep。
- Reader 的 x 轴是编码文件字节，其他图像算子是解码字节；PNG/JPEG 与文件大小口径不同，
  Reader 系数不可与算子系数直接相加比较。
- Batcher 的 sweep 是 batch 4、profile 里是 batch 1，计时方法也不同，两边系数不可直接比较。
- sweep 测的是 worker 内算子计算耗时，不含跨 boundary 传输、actor 启动与排队；跨后端结论要另外用
  `physical_model.boundary`（Ray payload 512 B – 4 MiB）等测量支撑。
- Gray/Blur/Jitter 等算子的固定开销占比很高（`fixed_fraction` 大），只按字节等比例外推（原 Cedar 的
  `y = x`）会显著低估；这正是 kx+b 拟合的动机。
- 既有快照不可改写（遵循 `AGENTS.md`）：需要新口径的曲线时另跑新实验，不要回填历史 profile。
- `evaluation/pipelines/simclrv2_multimodal/` 的生成脚本已在 `f98ff0e`
  （refactor: remove the fictional simclrv2_multimodal workload）中从工作区移除，
  第 4 批数据的可执行快照保留在 `outputs/iter_clip/modules/evaluation/pipelines/simclrv2_multimodal/` 等处。

## 6. 复现

```bash
# 容器内，先 source env/bin/activate
# 1) 原始尺寸 sweep（长实验，正式跑用 nohup；会先跑 gate 再跑 results）
bash outputs/target_pipeline_scaling_20260910/run.sh
# 2) 仿射拟合对照：重读上面的 results + cm/dp 两份 profile，重写 coefficients.csv/comparison.png
python -m evaluation.pipelines.target_pipeline.compare_simclr_affine
# 3) 逐算子曲线图（按 backend/modality 分页）
python -m evaluation.pipelines.target_pipeline.plot_latency_scaling --help
```

第 2 步依赖 `outputs/cm_vs_cedar_simclr_20260911/profile.yaml` 与
`outputs/dp_vs_old_dp_simclrv2_20260911/profile.yaml`，脚本会断言两者一致
（`profiles_identical: true`，见该目录 `metadata.json`）。
