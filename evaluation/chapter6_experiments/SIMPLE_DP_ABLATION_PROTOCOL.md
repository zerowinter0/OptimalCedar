# Simple-DP 当前实验协议（2026-09-18）

当前只保留四个 Simple-DP 方法：

| 名称 | 计划搜索 | compute profile | boundary | W / width |
|---|---|---|---|---|
| `simple_dp_boundary` | subset DP | 新的 isolated layered profile | fixed + bytes/bandwidth | 不联合搜索 |
| `simple_dp_workers_boundary` | subset DP | 新的 isolated layered profile | fixed + bytes/bandwidth | 对每个 W 的资源切片重新生成计划，stage width=1 |
| `simple_dp_workers_width_boundary` | subset DP | 新的 isolated layered profile | fixed + bytes/bandwidth | 联合比较 W 和 stage width，使用 measured scaling curve |
| `old_dp_boundary` | 与 `simple_dp_boundary` 相同 | Cedar 旧 whole-pipeline offload profile，经 Amdahl 反推 | fixed + bytes/bandwidth | 不联合搜索 |

三个当前方法严格不读取 `offloads.*.*.throughput` 来计算算子成本。本地算子使用
`physical_model.operator_affine` 中每个算子测得的 `kx+b`；Ray/SMP 算子使用
`offloads.*.*.backend_compute`，只按同一条 affine 曲线缩放到候选输入大小。其中
width 版本还读取 `physical_model.scaling`。profile 缺少这些新字段、或某个算子没有
测得 kx+b 时直接报错，避免静默退回 Cedar 的按字节计价；未能拟合的算子会列在
`physical_model.operator_affine.unfitted_operators` 供排查。算子计算量不再区分
per-data / per-record 标签。

`old_dp_boundary` 是对照组：它保留原有 Simple-DP boundary 实现，只读取 Cedar
的 baseline 和 whole-pipeline offload throughput，不读取 `backend_compute`。
一份 dual profile 同时保存两套观测，因此四个方法共享同一次 profiling，数据输入
和机器状态一致。

实验矩阵还保留 `dj-cedar-opt`、`pecan-cedar-opt`、`plumber-opt`、`cedar-opt`、
`ray-opt` 和 `unopti` 作为外部基线，不运行 `dp-opt`。每个负载只运行一轮；单个
cell 的进程墙钟上限为 3,600 秒。已知 Cedar 在 LLaVA pretrain 和 StackExchange
的优化会超时，因此这两个 cell 按用户要求跳过。

六个负载为 SimCLRv2、SimCLRv2-cache、CommonVoice 15,000 条、COCO 5,000 张、
LLaVA pretrain 1,000 条和 StackExchange 2,000 条。每个负载重新采集 profile，
不复用历史 profile。profile 使用一个 local worker、每个 Ray/SMP stage 一个
actor/process；dual 协议先保留 Cedar whole-pipeline 测量，再采集 isolated
operator、boundary、affine 和 width-scaling 数据。

启动脚本为 `run_layered_simple_dp_full_20260918_v2.sh`，默认输出目录为
`outputs/simple_dp_layered_fresh_all_20260918_v7`。layered input snapshot 在全部
legacy 测量完成后使用独立遍历采集，避免序列化和额外缓存预热污染 Cedar profile。长实验通过
`nohup` 运行，完整
保存 profile、计划、结果、日志、代码快照、metadata 和状态文件。

### 2026-09-18 CommonVoice rerun without CPU reserves

CommonVoice uses 15,000 inputs, one round, fresh dual-layer profile, and a 7,200-second per-cell limit. Both local and Ray per-worker runtime CPU reserves are removed from optimizer and execution accounting; local workers and actual SMP/Ray stage widths are still charged. Simple-DP variants and Plumber run first. Previous 300,000-input results remain preserved. Run: outputs/commonvoice_15000_no_reserve_20260918/matrix.

### Shared transport capacity in W-aware boundary variants

Only simple_dp_workers_boundary and simple_dp_workers_width_boundary use the shared-transport objective. DP retains additive compute C, additive existing boundary communication H, and additive shared Ray transport B as separate Pareto coordinates. B sums (input_bytes + output_bytes) / boundary_throughput for every RAY/TF_RAY stage; fixed latency stays in H. For each W the internal objective is C + max(H, W * B), and the aggregate comparison score is C / W + max(H / W, B). Thus only communication is subject to the shared-capacity maximum, and compute is added afterwards. SMP is excluded from B. The existing synchronous boundary calibration provides an effective transport capacity, not a separately measured physical network bandwidth; this is a capacity floor, not a complete contention model. Every feasible W is searched with its own resource slice and cost model; cross-W plan reuse, grouping by identical resource slices and W=1 lower-bound pruning are removed. Old run snapshots and results retain their original objective.

### Superseding communication objective: fixed divided by W, bytes unscaled

The two W-aware boundary variants now use (compute + sum(fixed_latency / submit_batch_size)) / W + sum((input_bytes + output_bytes) / boundary_throughput). Byte terms are unscaled for both Ray and SMP; no max between communication coordinates remains. Each W is independently searched. CommonVoice 15,000-record rerun compares only simple_dp_workers_boundary and simple_dp_workers_width_boundary, one round, 7,200 seconds per cell, reusing outputs/commonvoice_15000_no_reserve_20260918/matrix/commonvoice/profiles/shared.yaml exactly. New results: outputs/commonvoice_15000_w_boundary_unscaled_bytes_20260918.

### Mandatory remote Ray calibration and execution

Evaluation entry points enable CEDAR_RAY_REQUIRE_REMOTE=1 and default to the
cedar_remote placement resource for every Ray stage. Boundary actors now use
the same actor options, validate their actual location, and include 16 MiB
payloads. Cache schema v3 records driver and eligible remote node IDs/IPs.
Ray calibration failure stops profiling and writes raw observations to the
diagnostics directory. New experiment runners and comparison entry points
reject historical profiles without a valid schema-v3 remote boundary model;
they cannot silently substitute the 10 GB/s compatibility constant.
Historical profiles/results remain immutable; refreshing only the Ray boundary
section in a new profile copy is permitted with explicit provenance.
