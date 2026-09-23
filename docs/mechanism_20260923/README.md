# SimCLRv2 B/H/J 融合与卸载机制实验（2026-09-23，v2 修正版）

论文 3.1 第二张机制证据图的数据来源。问题：Cedar 把卸载收益归入算子成本、再对整块施加
融合 I/O 折扣 `rho = IO_fused / IO_base`，这个简化能否表示计算与数据交接的真实变化。

> **v1 → v2 修正**：v1 的 `service_raw.csv` 把"每批端到端时间"与"每记录成员时间"直接相减，
> 口径不一致；上层汇总又对已经是每记录的量再除了一次 batch size。原文件已留档在
`legacy_v1/`（附 `INVALID.md`），v1 的"计算 2.54 / 余项 8.02 ms/record"等数字**作废**。
v2 一律先在**批级**求和、再相减，每记录只除一次实际记录数，并逐批校验
`计算 + 其他开销 = 总时间`（`identity_error_ms_per_batch_max ≤ 1.4e-14`）。

## 1. 实验配置与版本

- 固定顺序（Cedar 在 SimCLRv2 上选中计划的顺序）：
  `ImageReader(8) → Grayscale(3) → RandomResizedCrop(6) → [B(2) H(5) J(4)] →
  to_float(7) → Normalize(1) → Batcher(0)`；只改 B/H/J 的组织：
  L-U / L-F = local 三独立阶段 / 一个融合阶段；R-U / R-F = Ray 三 actor / 一个融合 actor。
- 真实实现：`InProcessMapperPipeVariant`、`InProcessFusedOptimizerPipeVariant`、
  `RayMapperPipeVariant`、`RayFusedOptimizerPipeVariant`；R-U 的段间结果经 driver 转发
  （与 runtime 相同，无 actor→actor 直连）。
- 随机性：`seed = 20260923 + record_id*1000003 + offset(op)`；四组输出**逐位相同**
  （`service_verification.json`：`max_abs_diff = 0.0`，shape `(1,244,244)` uint8）。
- 输入：从真实流水线捕获 400 条进入 B 前的记录，序列化 59,944 B/条
  （`inputs/block_inputs_meta.json`）；`inputs/block_inputs.pt` 23 MB，仅保留在 `outputs/`，
  sha256 见 `MANIFEST.json`。
- profile：`outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml`
  （sha256 见 `run_manifest.json`）；代码 commit 见同文件与 `cedar_cost_breakdown.json`。
- 资源：driver 固定 CPU 12；远端 actor 固定 CPU 8/9/10（R-U）或 8（R-F）；
  `OMP/MKL/OPENBLAS/NUMEXPR=1`、`torch intra-op=1`；`cedar_remote` 放置资源 +
  `CEDAR_RAY_REQUIRE_REMOTE=1`；实测 actor 全部落在 **172.23.166.105**
  （`actor_placement_check.json`），`num_cpus=1`。
- CPU 拓扑（`cpu_topology.json`）：8/9/10/12 的 HT sibling 分别是 40/41/42/44，
  即**四个不同物理核**，governor=`schedutil`。

## 2. 实验 A（串行服务时间，无重叠；修正后）

主测量：每配置预热 120 batches、测量 766/774 batches（两轮独立运行），最快组 30.0 s；
另有 3 轮**交错顺序**（顺序随机轮换，记录在 `service_meta.json::run_order`）、
3 轮**同核**（R-U 三 actor 全固定 CPU 8）与"单独跑 R-U/R-F"三种控制。

单位 ms/source-record；`其他开销 = 每批端到端 − 该批成员墙钟之和`：

| 配置 | B | H | J | 计算合计 | 其他开销 | 总时间（主测量，2 轮） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-U | 9.7216 | 0.0461 | 0.4440 | 10.2117 | 0.3919 | **10.6036 ± 0.058** |
| L-F | 9.7220 | 0.0642 | 0.4496 | 10.2359 | 0.4374 | **10.6732 ± 0.032** |
| R-U | 14.0654 | 0.1270 | 1.1748 | 15.3671 | 16.3502 | **31.7173 ± 0.181** |
| R-F | 10.8831 | 0.0867 | 0.6239 | 11.5936 | 5.6397 | **17.2334 ± 0.361** |

**U/F 保留比例（fused / unfused）**：local 计算 1.0024 / 其他 1.1160 / 总 1.0066；
Ray 计算 0.7544 / 其他 0.3449 / 总 **0.5433**。

**控制对照（总时间 ms/record）**

| 配置 | 主测量 | 交错 3 轮 | 同核 3 轮 | 单独运行 |
| --- | ---: | ---: | ---: | ---: |
| L-U | 10.6036 | 10.5282 ± 0.167 | 10.5721 ± 0.212 | — |
| L-F | 10.6732 | 10.5281 ± 0.116 | 10.4735 ± 0.075 | — |
| R-U | 31.7173 | 35.0824 ± 0.985 | 36.2317 ± 1.584 | 29.7（120.5/117.4 ms 每批） |
| R-F | 17.2334 | 19.6022 ± 0.340 | 18.4669 ± 1.110 | 15.4（61.4/61.6 ms 每批） |

## 3. 融合后成员时间下降：查到什么、没查到什么

R-U 的成员时间系统性高于 R-F（B 14.07 vs 10.88、H 0.127 vs 0.087、J 1.175 vs 0.624），
且三个成员**同时**变快 → 不能只用"块内交接减少"解释。逐项排查（数据在对应文件）：

| 假设 | 检查 | 结论 |
| --- | --- | --- |
| 输入表示不同 | `operator_timing.csv` 逐事件记录 shape/dtype/stride/contiguity，四配置全部 `1x244x244`、uint8、contiguous、stride `59536x244x1`；输出逐位相同 | 排除 |
| CPU 绑定/HT/NUMA | 8/9/10 是不同物理核；同核控制（全绑 CPU 8）后 R-U 成员时间不变（15.37→15.31） | 排除 |
| 顺序/预热/轮次 | 3 轮交错、顺序轮换；每配置每轮预热 60 batches；每轮 blur 中位数 R-U 5.39/5.39/5.40、R-F 3.76/4.03/3.79 | 稳定存在，非顺序伪影 |
| 事件漏计/重复 | 每 (config, operator) 恰好 9,096 个事件、键 0..3031 无重复；CPU 时间≈墙钟时间 | 排除 |
| 测量包装进入计时 | 成对交错微基准：种子设置 +0.088/0.058/0.100 ms/次（在计时窗口外，落入"其他开销"）；计时+事件记录 +(-0.03)/+0.007/+0.020 ms/次 | 量级 <1%，不足以解释 |
| 空闲间隔（频率/调度） | 同一 actor 内成对交替：连续 4-call 突发 vs 每突发前 sleep 80 ms → blur 中位 4.11 vs 4.13 ms（0.99×） | 排除 |
| 输入张量"新鲜度"（新页） | 同一 actor 内成对交替：常驻张量 vs actor 内新建拷贝 → 2.61 vs 2.60 ms（1.00×） | 排除 |
| 其它 runner 存活 | 单独只跑 R-U/R-F（无 L-U/L-F、无另一个 Ray 配置）→ 比例不变（R-U 计算 60.8 vs R-F 41.7 ms/批） | 排除 |

**仍未确定**：在同一台远端机器上，"专用单算子 actor"与"融合 actor"里同一 callable 的
每记录时间差（约 1.3–1.4×）的**剩余原因**。已排除输入、绑定、顺序、事件计数、包装、空闲间隔、
张量新鲜度与并发 runner；剩余候选是 actor 进程级的执行上下文（例如融合 actor 在同一批内持续
分配/复用中间缓冲带来的分配器与缓存状态差异，或该节点在两种调用模式下的功耗/频率耦合）。
要判定需要硬件计数器（cycles/instructions、cache miss、频率）级测量，本实验未做。
**这不允许把差额写成"融合减少了计算"**；也不允许把它当成网络开销。

## 4. 交接成本能拆多细

同一次执行里可用的直接计时（`service_meta.json::path_timing`，单位 ms/sample）：

| 配置 | 每阶段 submit（序列化+提交） | 每阶段 get（排队+计算+回传+反序列化） |
| --- | ---: | ---: |
| R-U（3 阶段） | 0.92–1.03 | 0.49–0.58 |
| R-F（1 阶段） | 1.04–1.06 | 0.54–0.57 |

口径与可加性（重要）：

- submit 窗口 = 客户端组批 + 序列化 + `actor.process.remote` 调用；get 窗口 = `ray.get`，
  **包含 actor 内计算**、返回传输与反序列化，因此 **get 不能与 compute 相加**；
  `get − compute` 才是可归入交接的部分。
- R-U 每批 3 次 submit/get，合计客户端路径约 4.2–4.7 ms/record；"其他开销"实测
  16.35 ms/record，其余部分（≈12 ms/record）发生在 driver 侧组批、future/队列调度与
  进程间传递中，运行时没有更细的直接计时，**不做拆分**。
- "端到端 − 成员计算"在本文档统一叫**其他开销**，不叫网络时间。
- 未做 echo/空算子微基准；本次不需要它来支撑结论。若以后补做，必须单列，不得与真实算子
  拼成同一个堆叠柱。

## 5. Cedar 模型复算与原生/扩展溯源

`cedar_cost_breakdown.json`（全部由 `cedar/compose/optimizer.py` 的原实现导出）：

| 量 | 值 |
| --- | --- |
| baseline Q0 | 22.9795 ms/source-record（单 worker） |
| f_i（B/H/J） | 0.2621 / 0.0047 / 0.2915 |
| 基础成本（B/H/J） | 6.0228 / 0.1070 / 6.6984 ms/record（**声明顺序**下的输入尺寸） |
| Amdahl 阈值（B/H/J） | 1.3552 / 1.0047 / 1.4114 |
| RAY total_speedup | 1.661 / 1.184 / 1.590 → 全部 ≥ 阈值 → **clip 到 0** |
| SMP total_speedup | 1.693 / 1.120 / 2.148 → 同样 clip 到 0 |
| 目标顺序成员输入字节 | B/H/J 各 58.1 KiB/record |
| **IO_base / IO_fused（字节，目标顺序）** | 357,216 B / 119,072 B → **rho = 1/3** |
| IO_base / IO_fused（声明顺序） | 3,810,304 B / 952,576 B → rho = 1/5 |
| 目标顺序成员成本（INPROCESS） | B 1.5057 / H 0.0089 / J 0.5582 → 合计 2.0728 |
| 融合块成本（L-F） | 2.0728 × 1/3 = **0.6909** ms/record |
| 整计划成本 L-U / L-F / R-U / R-F | 10.5481 / 9.1662 / 8.4753 / 8.4753 |

> v1 把 `_calculate_cost_fused()` 的返回值（两个**成本**）标成了 "IO_base/IO_fused (MB)"。
> 它们是成本不是字节；真字节见上表（本版新增），并保留原成本口径。

**原生 vs 扩展（函数级逐字节比对，`cedar_cost_breakdown.json::provenance`）**

| 项 | 与首版（commit `f062305`）| 与 `optimizer.py.orig` | 结论 |
| --- | --- | --- | --- |
| `_calculate_cost_fused` | 相同 | 相同 | 融合 I/O 折扣公式**原生**，且**不区分后端**（吃的是已按后端定价的 cost map） |
| `_offload_and_fuse` / `_local_fusion` / `_fuse_local_smp` / `_fuse_tf` / `_enumerate_fusions` / `_calculate_offloads` | 相同 | 相同 | 后端枚举**原生**：RAY（非 TF，offload+fusion 开）、TF_RAY、TF、SMP（仅 `forbid_local_parallelism` 时）；**从不枚举普通 INPROCESS 融合** |
| `calculate_cost` | 已改 | 相同 | 后来加入：materialized fused node 分支、多组 `fused_pipes`、cache 前缀修正 |
| `_calculate_pipe_cost` | 已改 | 相同 | 后来加入 INPROCESS/None 早退；**Amdahl 反演与 clip 阈值未改** |
| `_is_optimizer_pipe` | 已改 | 相同 | 后来加入 `plan` 参数 |

四种配置的估价路径：

- **L-U**：原生路径（原优化器就在评估这种全 INPROCESS 计划，作为 baseline）。
- **R-U**：原生路径（原优化器就是逐个算子/逐切片枚举 RAY 卸载）。
- **R-F**：原生路径（cedar-opt 实际物化的就是单个 RAY 融合块）。
- **L-F**：**扩展计划**。折扣公式本身是原生的，但原搜索**不会**产生 INPROCESS 融合计划；
  给它定价依赖后来加入的 materialized-fused-node 分支。因此只能写
  "该公式可应用于本地融合计划"，不能写"原版 Cedar 会用该折扣选择本地融合"。

## 6. 实验 B（完整流水线，保留并发）

fixed plan、80,000 条/轮、3 轮 round-robin（原对照，未改动）：

| 配置 | W=1 (rec/s) | W=64 (rec/s) | 实测 Ray actor |
| --- | ---: | ---: | --- |
| L-U | 73.80 ± 0.33 | 2296.74 ± 21.43 | 0 |
| L-F | 73.80 ± 0.51 | 2343.59 ± 29.57 | 0 |
| R-U | 102.37 ± 1.98 | **N/A（3/3 失败）** | 3（W=1） |
| R-F | 96.77 ± 1.67 | 1195.32 ± 35.42 | 1（W=1）/ 64（W=64） |

- R-U@W=64 的失败原因：actor `num_cpus=1`，64×3=192 个 CPU 请求 > 远端 128 CPU →
  `Multiprocess dataset worker exited during startup`；actor-ready 超时提到 900 s 仍失败。
  **保持 N/A，不用降 W 或加资源的结果替代。**
- 实验 A 是单批在飞的串行服务时间，**不能取倒数当作实验 B 的吞吐预测**：
  R-U 在 A 是 31.72 ms/record（31.5 rec/s），在 B 的 W=1 实测 102.37 rec/s。

## 7. 三类结论（明确区分）

**A. 直接支持"计算与交接不能统一折扣"**

1. Ray 上 U→F：总时间 0.5433×，其中**计算 0.7544×、其他开销 0.3449×**——两者变化幅度差一倍以上，
   一个标量折扣（Cedar 本地 `rho=1/3`、声明顺序 `1/5`）无法同时表示。
2. Cedar 在 RAY 卸载上把三个成员全部 clip 到 0 → R-U 与 R-F 的**块成本相同（0/0，N/A）**，
   整计划成本也相同（8.4753 = 8.4753）；而实测串行服务时间差 **1.84×**（31.72 vs 17.23 ms/record）。
   这是"卸载块的计价把交接丢掉"的直接证据，不是排序噪声。
3. local 上 U/F 的实测差只有 0.7%（总时间），而模型折扣给出 0.3333×——
   折扣在"融合没有移除跨进程交接"时没有物理对应物。
4. 交接本身可测且显著：R-U 的"其他开销"16.35 ms/record（占总时间 51.6%），R-F 5.64（32.7%）；
   客户端直接计时的 submit 约 0.92–1.06 ms/sample/阶段（不含计算，可直接相加），
   get 含计算（不可与 compute 相加）。

**B. 只能说明"预测失准"，不能单独证明机制**

1. B 的 W=64：local 融合 2343.6 vs 单 actor Ray 融合 1195.3（1.96×），而模型偏好 Ray
   （9.1662 vs 8.4753）——排序反转，但 W=64 的 R-U 缺失、且 local/Ray 的 CPU 预算不同形，
   只能作为"模型排序不可靠"的证据。
2. 声明顺序 vs 目标顺序的 IO/成本差（rho 1/5 vs 1/3、成员成本 6.02/0.11/6.70 vs 1.51/0.009/0.56）
   说明 Cedar 的定价对算子顺序高度敏感；这是模型行为，不是对机制的直接测量。

**C. 仍未确定**

1. §3 表中列出的、已排除假设之外，R-U 与 R-F 成员计算差的**剩余原因**（见 §3 结尾）。
2. "其他开销"里约 12 ms/record 的 driver 侧调度/组批/future 框架成本的具体构成——
   运行时没有更细的直接计时，不做拆解。
3. 融合对算子**计算本身**是否有真实收益：本实验能确定的是"同一 callable 在不同后端上下文里
   时间不同"，无法在现有测量下把"真实计算变化"与"执行上下文差异"分离。

## 8. 产物清单（同一目录）

| 文件 | 内容 |
| --- | --- |
| `README.md` | 本文件 |
| `service_raw.csv` | 逐批：`elapsed_ms_per_batch`、`compute_<member>_ms_per_batch`、`compute_sum_ms_per_batch`、`other_overhead_ms_per_batch` 及每记录列、成员每记录列（v2 口径） |
| `service_summary.csv` | 逐轮汇总 + `identity_error_ms_per_batch_max` |
| `service_summary_repeats.csv` | 跨轮 mean±stdev |
| `operator_timing.csv` | 逐事件：wall_ms、cpu_ms、shape/dtype/contiguous/strides（本目录主测量的 10 MB 文件在 `control_interleaved/` 与 `control_samecore/`，清单见 `MANIFEST.json`） |
| `cedar_cost_breakdown.json` | 模型中间量、真 IO 字节、原生/扩展标识 |
| `pipeline_results.csv` | 实验 B 逐轮吞吐/状态/资源 |
| `figure_data.json` / `figure_data.md` | 可直接绘图的时间拆解、保留比例、吞吐对照 |
| `wrapper_overhead.json`、`burst_idle_probe_*.json`、`tensor_state_probe.json`、`cpu_topology.json` | 诊断（单列，不与主结果相加） |
| `legacy_v1/` | v1 原始文件与 `INVALID.md` |
| 控制目录 | `control_interleaved/`、`control_samecore/`、`repeat2/`、`probe_ray_alone/`、`instrument_overhead_{full,none}/` |

复现（容器内，先 `source env/bin/activate`）：

```bash
RUN=outputs/simclrv2_fusion_offload_mechanism_20260923
bash scripts/run_block_mechanism_20260923.sh                 # capture → A → B → 复算 → 图数据
python -u scripts/reaggregate_block_service.py --run-dir $RUN --include-subdirs   # v1→v2 恢复
python -u scripts/block_service_harness.py service --run-dir $RUN/control_interleaved \
    --configs L-U L-F R-U R-F --rounds 3 --warmup-batches 60 --min-batches 120 \
    --min-seconds 30 --cpu 12 --remote-cpu-base 8
python -u scripts/block_wrapper_overhead.py --run-dir $RUN --cpu 12 --calls 250
python -u scripts/block_burst_idle_probe.py --run-dir $RUN --repeats 40 --gap-ms 80 --remote \
    --out $RUN/burst_idle_probe_paired.json
python -u scripts/block_tensor_state_probe.py --run-dir $RUN --repeats 60 \
    --out $RUN/tensor_state_probe.json
python -u scripts/block_cpu_topology_probe.py --cpus 8 9 10 12 --out $RUN/cpu_topology.json
python -u scripts/block_mechanism_figure_data.py --run-dir $RUN
python -u scripts/sync_block_mechanism_artifacts.py --run-dir $RUN --dest docs/mechanism_20260923
```

仓库同步：小型数据文件与脚本已复制到 `docs/mechanism_20260923/`（见其 `MANIFEST.json`，
含大文件路径、行数与 sha256）。
