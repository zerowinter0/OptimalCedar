# Simple-DP 独立消融协议（2026-09-17）

本轮响应六负载实验要求，替代历史 W=8 协议。输出目录为
`outputs/simple_dp_ablation_20260917`。这是新实验，不覆盖 `final_result.md`
中历史筛查结果，不采用历史最佳值作为本轮结果。

## 方法

| 展示名 | 实现 | W 策略 |
|---|---|---|
| dj-cedar-opt | dj_optimizer | 该优化器自身策略 + 公共资源预算约束 |
| pecan-cedar-opt | pecan_optimizer | 同上 |
| plumber-opt | plumber_optimizer | 单输入流水线驱动，仅分配本机 stage 并行度 |
| cedar-opt | optimizer | Cedar 原有吞吐量阈值启发式 + 公共资源约束 |
| ray-opt | raydata_optimizer | 单执行器，远端 map stage 池 |
| unopti | UnoptimizedOptimizer | 单进程 INPROCESS，不重排、不融合、不缓存、不预取 |
| simple-dp-opt | SimpleDpOptimizer | Cedar 启发式 + 公共资源约束，不搜索 W |
| simple-dp+W | SimpleDpWorkersOptimizer | 独立遍历预算内全部整数 W |
| simple-dp+boundary | SimpleDpBoundaryOptimizer | 与 simple-dp 相同规则 |
| simple-dp+variant-max | SimpleDpVariantOptimizer | 与 simple-dp 相同规则 |
| simple-dp+width | SimpleDpWidthOptimizer | 先取得 simple-dp 实际 W，再保持 W 搜索 stage 宽度 |

不执行 dp_optimizer。四项是**分别单独增加**，不是逐项累计。

- simple-dp 的算子成本调用 `Optimizer._calculate_pipe_cost`，融合使用 Cedar
  的 fused-I/O / unfused-I/O 比例，所有算子标量求和，缓存使用 Cedar 磁盘读成本。
  关闭 W 搜索入口；原先虽然继承 DP 类，但覆盖的主搜索函数并没有实际调用
  DP 的 W 搜索，历史 simple-dp 的 W=21/32 不能直接证明发生了 W 搜索。
  同时绕过 affine 初始化的尺寸比例改写，去除共享 DP 中写死 PICO 目标的
  incumbent、suffix bound、block pruning 和 batch skyline 快速路径；以实际
  Cedar/消融目标进行通用精确 DP 转移和状态支配判断。
- W 消融不引入 PICO 的 contention / affine / bandwidth 等额外模型。
  Cedar 的单副本 DP 目标与 W 无关，因此 DP 只需求解一次，再遍历 W 计算
  `C_cedar / W` 并检查本机、远端的预算可行性。重复求解相同 DP 不增加搜索空间。
- boundary 消融取消乘法 I/O 折扣，保留 Cedar 算子成本，给每个实际远端/SMP
  stage 增加 `fixed_latency / submit_batch + (input_bytes + output_bytes) / bandwidth`。
  融合块只付一次边界。INPROCESS 不付跨后端边界。使用共享 profile 的 boundary
  校准，不启用 object-boundary、额外 selectivity、inflight、affine 等模型。
  该项测试的是“Cedar 算子成本 + 显式边界”的替代评分；原始 offload profile 是
  整管线测量，不应将这一消融解释为已经精确分离计算和通信的物理模型。
- variant-max 仅将聚合方式改为 `max(sum(INPROCESS/TF), sum(RAY/TF_RAY), sum(SMP))`。
  不按硬件 GPU 属性另分类，不引入 PICO 的 lane exposure、SMP 特殊处理或 GPU 倍率。
- width 仅增加 DP 的整数 actor/process 分配及预算状态，使用 `C_cedar / width`。
  不引入 PICO 拟合的平方根 fanout 或测量宽度上限；这些属于额外代价模型假设。
  先计算 simple-dp W 的时间也计入该项优化/构建时间。最终执行保留 DP 选中的宽度。

## Ray / Plumber 移植边界

[Ray Data 官方实现说明](https://docs.ray.io/en/latest/data/data-internals.html)
采用执行器、map fusion、任务/actor 调度和背压。移除原 Cedar Ray 基线中自行设置
8 个驱动进程的扩展。保留声明顺序和可融合的连续 map；不能融合的可并行算子
单独成为 stage，固定重排位置不再误当成禁止并行。

[Plumber 原论文](https://proceedings.mlsys.org/paper_files/paper/2022/hash/d0e90e9a9310570dfa643aa3b2da6e89-Abstract.html)
在主机资源约束下调节 stage parallelism、prefetch 和 caching。移除原移植中
`W * min(stage_rate)` 的额外 W 搜索及 Cedar 的驱动复制启发式。固定位置算子
仍可并行，是否能并行由后端支持决定。

两者均为 **Cedar 运行时中的策略移植**。Ray 移植采用静态 actor 配额，没有复刻
原生 Ray Data 的动态块调度、零拷贝对象引用路径和 actor autoscaling；Plumber
移植是离散 stage-rate 分配，没有复刻 tf.data tracer 和其内存 cache 重写器。
因此结果不能标为“原生 Ray Data / Plumber 端到端系统结果”。不为 Plumber
擅自增加原系统不具备的远端 Ray offload。

## 数据与资源

| 负载 | 输入范围 | batch 参数 |
|---|---|---:|
| SimCLRv2 | 已下载 Imagenette train 全集，条数写入 metadata | 4 |
| SimCLRv2-cache | 同一 Imagenette train 全集 | 4 |
| CommonVoice | 同一目录前 300 条，统一 max_samples=300 | 1 |
| COCO | val2017 全部 5,000 张 | 1 |
| LLaVA pretrain | 原 20,000 条 manifest 前 1,000 条 | 1 |
| StackExchange | 原 10,000 条 manifest 前 2,000 条 | 1 |

不改变算子、过滤阈值、图像大小、模型或增强次数。数据量参考 `final_result.md`
及 `tmp_analysis/workload_env.sh`，JSONL 子集固定并记录 SHA256。
COCO 历史“20,000 samples”是 5,000 次无 batch 输出乘以 batch_size=4；本轮
按 5,000 条真实输入计算。CommonVoice 等无末端 Batcher 的负载用 batch_size=1。

本机 172.23.166.103：CPU 预算 64；远端 172.23.166.105：Ray CPU 预算 64，
连接 `172.23.166.105:6379`，Ray stage 使用 `cedar_remote` placement。
不传固定 W，不继承外部 CEDAR_* 模型调参环境变量。

每个负载采集一份新的共享 profile：一个本地驱动、Ray stage 一个 actor、SMP
stage 一个 process，计时阶段 10 秒，另测固定延迟和吞吐量边界。profile 独立于
每个 cell 的 1 小时限制，profile 上限 3 小时；失败则跳过该负载并保留失败记录。
不复用先前挑选出来的 profile。

## 执行和计时

- 各负载顺序执行，各方法每轮 rotate 一位，三轮 round-robin，最多 198 个 cell。
- 每轮每方法重新优化。一个 cell 的进程墙钟硬上限 3,600 秒，包括启动、优化、
  构建、cache 预热、正式遍历和清理，比只限制优化+正式运行更严格。
- 首轮超时则记录 timeout，并在后两轮写 skipped_first_round_timeout；不会把超时
  样本删掉后再给“成功三轮均值”。其他失败保留具体日志和返回码。
- cache 仅对 SimCLRv2-cache 允许。每轮先清理该方法 cache，再完整预热，验证
  shard 提交完成后正式测量。预热不计入报告吞吐量。
- `num_total_samples=0`、完整遍历固定输入，防止提前截断留下未排空的工作。
- 主指标：固定输入条数 / `workload_wall_time_sec`，包含流水线启动、填充及排空，
  排除优化和预热。另保留框架原始 perf_time、输出计数、优化/构建耗时。
  SimCLR 原始 batch 计数可能含末尾补计，不用于主吞吐量。
- 结果输出 `RESULTS.md` / `summary.csv`，报告各轮状态、均值与样本标准差。
- 每个负载保存 profiles、plans、results、warmup_results、logs、cache；根目录
  保存代码快照、代码 SHA256、metadata、status.json、runner.pid 和总日志。
- 根目录已有时不覆盖。正式执行使用准备好的冻结代码副本，远端通过同一
  Ray runtime_env 分发，防止本机与远端代码不同步。

启动命令（在 optimalcedar-torch201-dev 容器内）：

```bash
cd /workspace/OptimalCedar
source env/bin/activate
nohup python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
  --output outputs/simple_dp_ablation_20260917 --prepared \
  > outputs/simple_dp_ablation_20260917/runner.log 2>&1 < /dev/null &
```

本次 smoke 使用历史 profile，仅验证代码和执行入口；这些数字不进入正式统计。
