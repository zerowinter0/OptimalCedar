# PICO DP 最优性验证：整数规划对照（2026-09-21）

目标：证明 `SimpleDpWorkersWidthBoundaryOptimizer`（PICO，selector 27）的 DP 确实求到了**它自己 cost
model** 的最优解。做法是把它搜索的**全部状态转移图**抽出来，交给独立 MILP 求解器求 min-max 最短路径，
再和 DP 报出的 cost 对比；MILP 选出的计划还会被 optimizer 自己的打分函数 **replay 复核**（可执行性 +
代价），避免"一个不可实现的更低解"造成假阳性。

工具：`scripts/verify_pico_dp_optimality_ilp.py`

- 用 `CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS` 固定 W、`CEDAR_DP_WORKER_SEARCH=0` 关掉 W 搜索、
  `CEDAR_DP_SEARCH_MODE=general` 强制通用 Pareto DP；
- 给所有 DP 搜索类打补丁，记录**每一次被评估的转移**（把 upper-bound pruning 关掉），每条边记录
  lane 增量（用 optimizer 自己的 `_accumulate_objective` 得到）、block 的顺序/后端/宽度与目标状态资源；
- MILP：弧变量 ∈{0,1}、流守恒（起点注入 1 单位，只有 full-mask 状态可以终止）、每条 lane 写成增量的
  线性函数、`min T s.t. T ≥ lane`，用 HiGHS（`scipy.optimize.milp`）求解。

PICO 的目标坐标是按 lane 相加、标量分数取 `max(lane)`（`_LANE_EXPOSURE` 默认全 0），因此在增量可加的
情形下这个 MILP 就是原问题的精确编码。

## 结果

| 实例 | W | search runs | nodes/edges（每个 run） | DP cost | MILP 最优 | MILP 计划 replay | 结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| commonvoice | 64 | 63 | 120 / 1,440 | 27.070672 | **27.070671591737725** | 27.070672 | **DP = MILP，DP 最优**（差 3.6e-15） |
| simclrv2 | 64 | 64 | 1,003 / 14,343 | 8.205709 | **8.205709** | 8.205709 | **DP = MILP，DP 最优**（差 0） |
| commonvoice | 32 | 63 | 204 / 1,776 | 26.432393 | 25.776682 | ✗ `ValueError: SMP transport curve does not cover requested W` | 不可判定（见下） |
| simclrv2 | 32 | 64 | 1,673 / 29,270 | 7.271867 | 8.205709 | 8.205709 | 不可判定（见下） |

W=64 正是 campaign 里 PICO 实际产出计划的那组配置（commonvoice 全融合 local、simclrv2
`Fused{2,5,4,3,6,1,7}`），两个实例上 MILP 的最优值和 DP 报出的 cost **逐位相同**，而且 MILP 选出的计划
被 optimizer 自己 replay 出同样的数值 —— 即：在这两个实例上，DP 的 Pareto 剪枝、状态合并与最短路径选择
没有丢掉任何更优解，DP 求到的就是该 cost model 在这个候选空间上的最优解。

## W=32 的两个不一致（值得修，但不影响上面的结论）

1. **search 与 replay 的可行性口径不一致**：commonvoice W=32 时，MILP 找到 25.7767（比 DP 的 26.4324
   低 2.5%），但把它交给 optimizer 自己的 `_replay_dp_objective` 会抛
   `ValueError: SMP transport curve does not cover requested W` —— 说明搜索阶段给 SMP stage 定价时用的
   worker 数与"物化计划后再打分"时用的 W 不是同一个，聚合传输曲线覆盖不到后者。搜索因此可能给一个
   实际不可行的 SMP 计划报出偏低的 cost。
2. **目标坐标并非处处可加**：simclrv2 W=32 时 DP 报 7.2719，而"全部被评估转移"的图里最优只有 8.2057
   （即 DP 报出的那条路径在该图里复现不出来）。`SimpleDpWorkersBoundaryOptimizer` 里存在
   `_SharedCommunicationObjective` 这类"共享/聚合通信"语义（一条记录只在全局计一次），当计划被拆成多个
   block 时它不是简单求和，因此本 MILP 的线性编码在该区间不精确；要用大 M / 指示变量的分段编码才能覆盖。

也就是说：**在最优点由单块融合计划给出的配置（W=64，也正是我们实验用的配置）上验证通过；W≠64 的实例
暴露出搜索期成本与物化打分的两处口径不一致，需要单独修**，这两点已在上面列明。

## 复现

```bash
# 容器内
python -u scripts/verify_pico_dp_optimality_ilp.py \
  --workload commonvoice \
  --profile outputs/ultimate_eight_optimizers_20260920/commonvoice/profiles/shared.yaml \
  --workers 64
python -u scripts/verify_pico_dp_optimality_ilp.py \
  --workload simclrv2 \
  --profile outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml \
  --workers 64
```

运行日志：`outputs/pico_dp_ilp_verification_20260921/`。
