# Cedar 融合折扣 `C_fused = ΣC_member × ρ` 的针对性实验（2026-09-23）

检验：`ρ = IO_fused / IO_unfused` 这个 I/O 比例能否直接作用于整个融合块成本。
固定顺序、输入、后端、并发与计时边界，只改融合边界；用可调计算量覆盖
"边界占主导 → 成员计算占主导"。**不研究 reorder，也不改论文。**

## 一、候选真实块扫描（纯定价，无执行）

`candidate_blocks.json`（脚本 `scripts/scan_fusion_block_candidates.py`）。声明顺序下
7 个 mapper 的 Ray 成本：

| 算子 | 声明输入字节 | Ray 成本 (ms/记录) | 是否被 clip 到 0 |
| --- | ---: | ---: | --- |
| to_float(7) | 597,552 | **14.8674** | 否 |
| RandomResizedCrop(6) | 2,390,210 | **11.1598** | 否 |
| RandomHorizontalFlip(5) | 714,432 | 0 | 是 |
| ColorJitter(4) | 714,432 | 0 | 是 |
| Grayscale(3) | 714,432 | 0 | 是 |
| GaussianBlur(2) | 238,144 | 0 | 是 |
| Normalize(1) | 238,144 | 0 | 是 |

可用连续块：`to_float→crop`（两成员全非零，ρ=0.2153，Σ=26.0273，块成本 5.6049）、
`to_float→crop→flip`（前两成员非零、第三成员被 clip，ρ=0.1744，Σ=26.0273，块成本 4.5401）。
更长的块只能追加零成本成员。**B/H/J 块（全零）仅作机制对照，不用于检验折扣本身**（按要求）。

## 二、实验 A：受控三算子块（ρ 固定 = 0.5）

三算子 A→B→C，输入输出同 shape/dtype/布局（float32 1×244×244，238,144 B/条）；
每个算子把同一个确定性 conv+clamp 重复 K 次后返回最后一次结果（**输出逐位一致**，
`max_abs_diff = 0.0`）。U = 3 个独立 Ray 阶段，F = 1 个融合 Ray 阶段，P = 融合 A+B + C 独立。
3 轮交错顺序、每轮 120 批（4 条/批）、预热 40 批、所有 actor 绑同一物理核（CPU 8）、单线程。

| K | T_U | T_F | 实测保留 T_F/T_U | 成员计算保留 | 其他开销保留 | U 成员占比 | 规则 T_U×ρ | 规则/实测 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.274 | 16.027 | 0.430 | 0.905 | 0.334 | 0.168 | 18.637 | 1.163 |
| 4 | 53.773 | 22.745 | 0.423 | 0.661 | 0.289 | 0.361 | 26.886 | 1.182 |
| 16 | 84.625 | 45.982 | 0.543 | 0.724 | 0.268 | 0.604 | 42.313 | 0.920 |
| 64 | 187.997 | 170.583 | 0.907 | 1.055 | 0.299 | 0.805 | 93.999 | 0.551 |

（单位 ms/记录；T_U/T_F 为 3 轮均值，逐轮标准差见 `expA/expA_summary.csv`。）
P@K=16：62.922（保留 0.743），ρ_P = 2/3，规则预测 56.4 → 规则/实测 0.897。

**结论（A）**：ρ 恒为 0.5，实测融合保留随计算占比从 **0.430 涨到 0.907**；
规则预测与实测之比从 +16% 漂到 −45%。**即：单靠 ρ 无法决定融合收益**。
注意成员计算保留并非恒 1（0.66–1.05），见第四节。

**插桩对照**（轻/重两档，关闭成员计时、只留批端到端时钟，1 轮）：

| 档位 | 有插桩 | 无插桩 | 差异 |
| --- | ---: | ---: | ---: |
| U@1 | 37.274 | 37.683 | +1.1% |
| F@1 | 16.027 | 16.843 | +5.1% |
| U@64 | 187.997 | 190.339 | +1.2% |
| F@64 | 170.583 | 141.930 | **+20.2%** |

即重档的融合 cell 受插桩影响明显（原因未定）；用无插桩数值时 K=64 的保留为
141.93/190.34 = **0.746**，仍远高于 ρ=0.5，主结论方向不变。

## 三、实验 B：真实块 `to_float → RandomResizedCrop → RandomHorizontalFlip`

声明顺序、原始算子参数、Ray 放置不变；输入为真实 ImageReader 输出（uint8，
中位 542,671 B/条；shape 随图片变化）。同样的串行协议（3 轮、120 批、同一物理核）：

| 组织 | 总时间 | 成员计算 | 其他开销 | actor | 保留 vs U |
| --- | ---: | ---: | ---: | ---: | ---: |
| U（3 个 Ray 阶段） | 121.539 | 6.825 | 114.714 | 3 | 1.000 |
| P（融合 7+6，5 独立） | 51.558 | 6.687 | 44.872 | 2 | 0.424 |
| F（融合 7+6+5） | 28.314 | 6.564 | 21.749 | 1 | **0.233** |

Cedar 模型：成员 Ray 成本 14.8674 / 11.1598 / 0（被 clip），Σ=26.0273；
真实字节 IO_base 7,521,267 → IO_fused 1,311,984，**ρ = 0.1744**；公式块成本 4.5401。
**规则检验**：T_U × ρ = 21.20 vs 实测 28.31（规则/实测 = **0.749**，低估 25%）。
成员计算几乎不变（保留 0.96–0.98），被省掉的是交接（其他开销保留 0.19–0.39）。

## 四、实验 C：完整流水线（W=1，块外计划完全不变）

只改该块的边界（U/P/F），块外顺序与算子全部保持声明计划；4,000 条/轮 ×3 轮：

| 计划 | Cedar cost | 块内 cost | 块外 cost | 预测加速 | 实测吞吐 (rec/s) | 实测加速 | actor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U | 47.1159 | 26.0273 | 21.0886 | 1.000 | 17.35 ± 0.47 | 1.000 | 3 |
| P | 26.6935 | 5.6049 | 21.0886 | 1.765 | 46.99 ± 1.14 | **2.709** | 2 |
| F | 25.6287 | 4.5401 | 21.0886 | 1.838 | 52.64 ± 1.89 | **3.034** | 1 |

块外模型贡献三者完全一致（21.0886，`block_external_identical = true`），
所以差异只来自块内计价；预测方向正确但幅度低估约 1.5–1.65×。
注意：该实验保留运行时阶段重叠，融合同时改变 actor 数与跨阶段并行结构；
**串行服务时间的倒数不能当作流水线吞吐预测**（实验 A 才用于隔离 I/O 折扣）。
W=64 与"多算子独立 offload 组合误差"本轮不做。

## 五、结论三类

**1. 直接支持"整体 I/O 折扣不成立"**

- 实验 A：ρ 固定 0.5，实测融合保留 0.430→0.907（成员占比 16.8%→80.5%），
  规则预测偏差 +16%→−45%；其他开销保留稳定在 0.27–0.33，成员计算保留 0.66–1.05。
- 实验 B：ρ=0.1744，实测 F 保留 0.233（规则低估 25%）；P 保留 0.424 也远高于其自身 ρ_P=0.2153。
- 两者共同说明：融合保留 ≈（保留的成员计算）+（保留的交接），而 ρ 只描述交接的一部分，
  且成员计算并不会按 ρ 缩小。

**2. 只说明端到端估价不准**

- 实验 C：预测加速 1.765/1.838 vs 实测 2.709/3.034（低估 ~1.5–1.65×）；
  由于块外贡献完全相同，误差全部来自块内定价（U 的"三成员各自 clip 后相加"与
  P/F 的"Σ×ρ"两种写法都没有对应实测）。
- 该结果本身不能定位是 ρ、还是 clip、还是 actor 并行结构导致，需要实验 A/B 的隔离证据。

**3. 尚未解释的成员执行时间变化**

- 实验 A 中成员计算保留随 K 变化（0.66/0.72/1.05），K=64 时融合 actor 的成员计算
  反而比未融合高 5%；插桩关闭后 F@64 的端到端下降 20%（U@64 只降 1.2%）。
  这与 §4.7 观察到的"融合 actor 里成员时间下降"方向相反，说明成员执行时间受
  执行上下文影响且方向不定；本实验未做硬件计数器级诊断，**原因仍未确定**。

## 六、产物与复现

| 文件 | 内容 |
| --- | --- |
| `candidate_blocks.json` | 候选块扫描（成员 Ray 成本、clip、真实字节、ρ、块成本） |
| `expA/expA_config.json` | 张量规格、K 档、ρ、配置定义、CPU 绑定 |
| `expA/expA_raw.csv`（逐批） / `expA_summary.csv`（逐轮） / `expA_operator_timing.csv`（逐事件） | 实验 A |
| `expB/...` | 实验 B（同结构） |
| `expC/plans/*.yaml`、`expC/scores/*.json`、`expC/throughput/*.json`、`expC/actor_probe_*.json` | 实验 C |
| `cedar_costs.json` | 成员 Ray 成本、真实字节、ρ、块成本、整计划成本、块外贡献 |
| `figure_data.json` / `figure_data.csv` / `figure_data.md` | 可直接绘图的数据 |
| `expA_instrument_off/` | 轻/重档位的无插桩对照 |

复现（容器内，先 `source env/bin/activate`）：

```bash
RUN=outputs/fusion_discount_20260923
python -u scripts/scan_fusion_block_candidates.py \
    --profile outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml \
    --declared-plan outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/plans/round1__unopti.yaml \
    --out $RUN/candidate_blocks.json
# 实验 A（受控块）
python -u scripts/fusion_discount_harness.py run --run-dir $RUN/expA --levels 1 4 16 64 \
    --partial-level 16 --rounds 3 --batches 120 --warmup 40 --cpu 12 --remote-cpu 8
# 实验 B（真实块；先用 scripts/capture_reader_outputs.py 准备输入）
python -u scripts/fusion_discount_harness.py run --run-dir $RUN/expB --block real \
    --levels 0 --partial-level 0 --rounds 3 --batches 120 --warmup 40 --label expB
# 实验 C（完整流水线 W=1）
python -u scripts/build_simclrv2_plan.py --order FCHJGBN --structure RAY:7 RAY:6 RAY:5 \
    INPROCESS:4 INPROCESS:3 INPROCESS:2 INPROCESS:1 --workers 1 --out $RUN/expC/plans/U.yaml
python -u scripts/run_fixed_plan_throughput.py --plan $RUN/expC/plans/U.yaml ... --num-total-samples 4000
python -u scripts/fusion_discount_cost_export.py --run-dir $RUN --block-pipes 7,6,5 \
    --declared-plan .../round1__unopti.yaml --plans $RUN/expC/plans/{U,P,F}.yaml --labels U P F \
    --out $RUN/cedar_costs.json
python -u scripts/fusion_discount_figure_data.py --run-dir $RUN
```
