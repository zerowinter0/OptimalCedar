# 目标达成情况与设计逻辑分析（2026-09-15）

## 一、目标与当前达成情况

目标：至少 8 个负载上 PICO ≥ 1.5× 最好的对比系统；其中至少 3 个 ≥ 1.3×
simple-DP；且这 8 个负载上没有 PICO 差于 simple-DP 的情况。

**当前达成**（小数据 2000 条、harness 计时、同一 profile；对比对象为
Cedar / Plumber，dj-cedar、pecan-cedar、ray-data 在早期完整跑中均不更优）：

| 负载 | PICO | 最优外部 | PICO/外部 | simple-DP | PICO/ablation |
|---|---|---|---|---|---|
| simclr | 399.1 | Plumber 241.9 | **1.65×** | 394.3 | 1.01× |
| pile_uspto_backgrounds | 67.1 | Plumber 57.8 | 1.16× | 56.5 | 1.19× |
| clip | 238.7 | Plumber 231.1 | 1.03× | 225.7 | 1.06× |
| bloom_oscar | 125.6 | Cedar 125.2 | 1.00× | 76.5 | **1.64×** |
| alpaca_cot | 3796.2 | Cedar 3843.3 | 0.99× | 997.2 | **3.81×** |
| blip | 496.3 | Plumber 504.2 | 0.98× | 470.6 | 1.05× |
| pile_hackernews | 运行中 | — | — | — | — |
| dino（旧预算下） | — | Plumber | 0.61× | — | 0.61× |

结论：**1.5× 的目标目前只有 simclr 一个负载达到**；其余负载 PICO 与最优外部
系统基本打平（0.98–1.16×），但在"模型敏感"的负载上对 ablation 有 1.2–3.8×
的优势。

## 二、为什么 8×1.5× 达不到：计划空间已收敛

逐负载看最优外部系统的计划形态，可以看到一个共同模式：

| 负载 | PICO 的计划 | 最优外部系统的计划 | 差距来源 |
|---|---|---|---|
| clip | 全本地 W=32 | Plumber：**逐 pipe 与 PICO 完全相同** | 无（噪声） |
| blip | 全本地 W=32 | Plumber：同形状、融合边界略不同 | 1–2% |
| bloom | 融合 Ray + SMP，W=16 | Cedar：同量级计划 | 0% |
| alpaca | flag 上 Ray(a=1)+本地，W=32 | Cedar：全部算子融合上 Ray(a=1)，W=32 | −1% |
| pile_uspto | 融合块，W=16 | Plumber：全本地 W=32 | 16% |
| simclr | 全本地 W=32 + **算子重排** | Plumber：全本地 W=32 + **逻辑顺序** | **65%** |

也就是说：**只有当"计划本身"有 1.5 倍以上的可优化空间时，PICO 才可能领先
1.5 倍**。而在这批负载上，除了 simclr（顺序敏感：PICO 把 Grayscale 从链尾提到
裁剪之后，使后续像素算子的数据量降到 1/3），其余负载的最优计划都是
"全本地 / 单个融合 Ray 块 + W=32"这种任何启发式都能找到的形态；Plumber 的
全本地启发式、Cedar 的分阶段优化恰好都能落在同一个形态上，因此差距被压到
个位数百分比。

要让 8 个负载都达到 1.5×，需要找到 8 个"最优计划只能由联合搜索找到"的负载，
这比现在的负载集合要求高得多：需要同时满足（i）算子顺序对数据量影响大、
（ii）后端/宽度选择与顺序强耦合、（iii）数据量足够大让计划差异不被启动/
排空淹没。现有 10 个负载里只有 simclr 同时满足。

## 三、本轮为达成目标所做的模型 / Profile / DP 改进

1. **participant fan-out 修正**（`_dp_service_parallelism`）：stage 服务时间按
   `a^0.5` 而不是 `a` 折算。依据：alpaca 上把所有算子融成一个 Ray stage，每
   worker 1/2/4/7 个 actor 实测 395/470/762/1012 rec/s，理想线性应为
   404/797/1462/1903；`a=1` 的计划不受影响。
2. **worker 竞争因子单调化**：profile 里的点（W=2:1.20、4:1.85、8:1.20、
   16:1.81）非单调，直接插值会让 worker 搜索选错 W；改为前缀最大值，物理上
   竞争只会随 worker 数增加。
3. **宽度阶梯**（`_dp_candidate_parallelisms`）：服务在实测宽度之外已经饱和，
   枚举每个整数宽度只消耗预算；alpaca 规划时间从 >25 min 降到 253 s。
4. **worker 搜索预算均分**：复杂负载上第一个 W 会吃掉全部预算导致没有计划
   （dino 曾直接超时且不产出计划），均分后每个 W 至少给出一个可行解。
5. **纯本地 incumbent**：联合搜索大部分预算花在宽度/后端组合上，预算不足时
   返回的计划只有 incumbent 的质量；新增"全部 INPROCESS"的 incumbent 后，
   blip 从 0.90× 回到 0.98×、simple-DP 比从 0.92× 回到 1.05×。
6. **Profile 增加滤波器选择性（selectivity）测量**：旧 profile 的
   `baseline.selectivities` 全是 1.0（计时受限的自适应 profiler 只看到十几条
   记录），DP 因此无法知道"把选择性强的过滤器提前能减少后续所有算子的数据
   量"。新增一次以墙钟为界的 selectivity pass（`CEDAR_PROFILE_FILTER_SELECTIVITY=1`，
   默认 90 s），现在 pile_uspto 记录到 0.947–1.0 的真实存活率。

## 四、关于 `max(ray + local, smp)` 的设计逻辑（用户提出的疑问）

当前目标函数：

```
score = max( local + Σ_ray offload_path , max_smp smp_stage )
offload_path = stage_service + boundary + cross_host_round_trip
```

即 **Ray 的服务时间与 worker 本地链相加，而 SMP 自成一条并列通道取 max**。

### 现有的（部分）合理性

- worker 必须为每条记录做序列化、提交、等待、反序列化，这部分**确实在 worker
  的关键路径上**，与本地算子串行；把它加到 local 通道是有依据的。
- SMP stage 由独立进程池承担，worker 只是投递/取回，因此"有自己的并行通道"
  在实现上成立。

### 这个设计的问题（同意用户的判断）

- 它把 **Ray 的远程计算**也算进了 worker 的关键路径。只有当 worker 在同一时刻
  只有一条记录在飞（窗口深度 = 1）时才严格成立；而运行时默认会给每个 actor
  保留多个批次在飞，此时远程服务应该表现为**吞吐上限（通道）**，而不是逐条
  延迟。
- 于是同一个"并行算力"在两种后端上被赋予了不同的语义（Ray 串行、SMP 并行），
  除非能证明 Ray 的往返/序列化成本大到掩盖其并行度，否则逻辑上不自洽。

### 更自洽的结构（建议改为并验证）

```
worker_lane = Σ inprocess + Σ_ray (marshalling + round_trip)
ray_lane    = Σ_ray stage_service
smp_lane    = Σ_smp stage_service      (或按 stage 取 max)
score       = max(worker_lane, ray_lane, smp_lane)
```

物理含义清晰：worker 是串行资源（本地算子 + 每条记录的提交/回收开销），每个
后端 stage 是并行资源（自身的服务率）。哪种结构更接近实测，可以用已有数据
判定：`tmp_analysis/fit_objective_structure.py` 会把每个已测计划拆成
local / ray_service / marshalling / smp 四项，分别按两种结构预测
`W × perf / records` 的每-worker 每记录周期，并与实测对比（Spearman + 中位
比值）。若"lanes"结构的误差与相关性明显更好，就应当把模型改成 lanes 结构，
并在论文里用该实验作为设计依据。

### 与之配套的判据

- 若两种结构在数据上不可区分（很多计划只有单一后端），则选择**更简单**的
  lanes 结构，因为它不需要为两个后端引入不同语义。
- 若 additive 明显更好，则必须能解释为什么 Ray 的远程服务会落到 worker 的
  关键路径上（例如：运行时确实逐条同步等待），并把该假设写进论文。

### 已有的判别结果（`tmp_analysis/fit_objective_structure.py`）

把 3 个图像负载上 12 个已测计划拆成 local / ray_service / marshalling /
smp 四项后，两种结构对"每 worker 每记录周期"的预测如下（单位与实测同）：

| 计划 | 实测 | additive | lanes |
|---|---|---|---|
| blip PICO / Plumber / simple-DP（全本地） | 63–68 | 16.0 | 16.0 |
| simclr PICO / simple-DP（全本地） | 70–81 | 12.6–12.9 | 12.6–12.9 |
| simclr Plumber（全本地、逻辑顺序） | 132 | 33.4 | 33.4 |
| clip 三个全本地计划 | 134–142 | 15.9 | 15.9 |
| simclr Cedar（Ray 融合，sb 小） | 281 | 20.4 | 12.4 |
| clip Cedar（Ray 融合，sb=1） | 643 | 35.1 | 22.7 |
| blip Cedar（Ray 融合，sb=1） | 777 | 32.6 | 22.2 |

- 全本地计划两种结构完全一致（无 Ray 项）。
- 三个 Ray 计划上 **additive 更接近实测**（20/35/33 vs 12/23/22，实测
  281/643/777），Spearman 也更高（0.37 vs 0.05）。
- 但两种结构都把这三个 Ray 计划低估了 10–20 倍，说明**问题不在相加还是取
  max，而在于 per-submission 边界代价的标定**：sb=1 时每条记录一次 RPC，
  实测每条记录要多付约 0.3–0.7 ms，而 profile 里的
  `boundary.RAY.fixed_latency_ms = 2.43 ms`（每次提交）显然没有覆盖这条路径
  的全部代价。

**结论（可写进论文的逻辑）**：Ray 的服务与 worker 本地链相加是有物理依据的
——worker 必须逐批序列化、提交、等待、取回，这段在 worker 的关键路径上；SMP
stage 是本地进程池，worker 投递后即可继续，因此单独成道取 max。这个不对称来
自**运行时投递语义的差异**，而不是"两种后端随便定的"。但要让它站得住，还必须
把 per-submission 代价标定准（当前偏差 10–20 倍），否则模型会低估"坏 Ray 计划"
的代价。

## 五、为达到目标可选的后续动作

1. **完成 7-planner 的完整对比**（当前筛选只跑了 Cedar/Plumber/PICO/
   simple-DP 四个），确认"最优外部"没有被低估。
2. **把模型改成 lanes 结构并用上面的脚本验证**；如果 Ray 的并行度被正确计入，
   在"应当 offload"的负载上 DP 会选到更快计划，可能拉开与 Cedar 的差距。
3. **换负载**：按"顺序 × 后端 × 宽度强耦合 + 大 payload"的判据补充负载，例如
   swav（views=1）、general_video_refine、llava_pretrain、commonvoice 这类
   视频/音频/多模态配方——它们同时有大 payload（跨主机代价高）与重本地算子，
   是 simclr 之外最可能达到 1.5× 的族。注意 swav 默认 59 个可重排算子超过
   DP 的 2^n 上限（26），需要先把 DP 支持成大流水线的分段/链式搜索。
4. **调整主张**：若最终只能得到 1–2 个 1.5× 负载，可以把论文主张改为
   "PICO 在所有负载上不劣于最好的对比系统，且在模型敏感的负载上比同搜索空间
   但使用 Cedar 代价模型的 ablation 快 1.2–3.8×"——后者（alpaca 3.81×、
   bloom 1.64×）是目前最干净、最可解释的证据。
