# 算子 affine 代价、加速器宽度credit 与 Ray lane（2026-09-16 上午实现）

本文记录本轮针对"让 cost model 真正决定计划"的三项实现：每个算子的 kx+b
拟合（§1）、加速器 stage 的宽度 credit（§2）、以及 59 算子负载的可用性修复
与 4-view 等价配置（§3），并给出 Ray lane 开关的实验计划（§4）。

## 1. 每个算子的独立 kx+b（`physical_model.operator_affine`）

**测什么**：对每个算子在"它实际见过的最小合法输入"和"最大合法输入"上各做一次
计时，拟合 `cost(record) = k · input_bytes + b`（ms/record）。

**为什么需要第二个尺寸点**：图像类 pipeline 在 resize 之后每条记录的 payload
形状完全相同，算子自身的输入没有尺寸差异；此时用它自己的合法输入**按类型缩放**
造第二个点（PIL 用 `Image.resize`、`torch.Tensor` 用 `interpolate`、文本截断或
重复、dict/list 递归），保持记录结构与真实数据一致，而不是混用别的算子的
快照（第一版这样做过：把 uint8 图喂给期望 float32 的算子，测出的斜率是负的，
已废弃）。

**存什么**：

```
physical_model.operator_affine:
  method: two_stratum_operator_affine_fit
  operators:
    4: {fixed_fraction: 0.070, k_ms_per_byte: 1.17e-05, b_ms: 0.626,
        x_reference_bytes: 714852, points_ms_per_byte: [[179024, 2.724], [2858151, 34.118]],
        source: rescaled_legal_input}
```

`fixed_fraction = b / (k·x_ref + b)` 就是"这条算子的代价里不随输入缩小的比例"。
实测 simclr：7/10 个算子拟合成功，`fixed_fraction` 从 0.07（尺寸敏感）到 0.90
（近似 per-record）分布，正好对应论文里"两类算子"的说法。

**DP 怎么用**：`_BlockCostIndex` 把每个算子的 profile 代价拆成两部分

```
cost(prefix) = (1 - f) · c · (w / w_ref) + f · c
```

`w` 是 DP 当前 prefix 的字节工作积，`w_ref` 是该算子 profile 位置的工作积。这样
在 profile 位置仍**精确复现**原值（不改变已有标定），而重排到更小 payload 之后
时，只有可缩小的 `(1-f)` 部分跟着变小——DP 因此不会再为了"缩小输入"而把一个
几乎全是固定代价的算子搬来搬去。

**只属于 PICO**：`DpOptimizer.uses_affine_operator_cost = True`；
`SimpleDpOptimizer`（Cedar 原模型 ablation）与 `DpTwoStageOptimizer` 显式设 False，
否则它们的搜索成本与上报成本会不一致（第一版实现就是这样触发了
`Materialized Cedar cost diverged from Simple DP search` 自检失败）。

## 2. 加速器 stage 的宽度 credit

旧模型把 CUDA block 的服务整条记在记录的关键路径上（"单卡串行服务需求"），因此
任何"多副本共享一张卡"的计划都被判为慢。但 profile 里本来就有该算子在
1/2/4/8 actor 下的实测曲线：

| llava 算子 | 1 actor | 2 | 4 | 8 | 8 actor 实测加速 |
|---|---|---|---|---|---|
| ImageTextMatching (BLIP) | 32.28 ms | 23.61 | 21.45 | 17.29 | 1.87× |
| ImageTextSimilarity (CLIP) | 21.00 ms | 16.47 | 8.09 | 6.05 | 3.47× |

现在 `gpu_serial` 用 `block.cost / min_operator_speedup(block, width)` 计算，加速比
直接取实测值（取块内最保守者），宽度 1 时行为不变。

## 3. 长 pipeline（30+ 算子）的规划可用性

三处一起修，才让 4-view 的多视图 recipe 在预算内规划出来（31 算子、7.8 MB/记录）：

1. `_calculate_data_size_ratio` 容忍 profile 里多出来的 pipe（SwAV 59 算子以前在
   setup 阶段抛 `KeyError: 59`）。
2. **确定性子集表改成按需计算**：`_calculate_all_non_random` 原本为所有 2^n 个
   mask 建表，n=31 时是 2^31 个条目——实测吃掉约 20 分钟和 17 GB，搜索还没开始
   预算就没了。现在是惰性记忆化（只用搜索真正访问到的 mask）。
3. **chain-partition 搜索的融合段长度上限**（`CEDAR_DP_CHAIN_MAX_SEGMENT`，默认 8）
   与 **DP 不再跑 Cedar 的枚举式重排 pass**（`skip_legacy_reordering`）——后者会
   枚举依赖图的全部合法重排，对已经在搜索顺序的 DP 完全冗余。

效果（swav_views4，31 算子）：

| | 修复前 | 修复后 |
|---|---|---|
| PICO | 8.8 rec/s（预算耗尽，只能返回 W=1 的 incumbent） | **129.0 rec/s（W=32）** |
| Plumber | 128.6 rec/s | 127.1 rec/s |
| 比值 | 0.07× | **1.01×** |

也就是说：修完之后 PICO 在这条重负载 recipe 上追平了最强 baseline（该负载
7.8 MB/记录，机器带宽/算力已饱和，1.5× 没有空间）。

## 4. Ray lane 开关（`CEDAR_DP_RAY_MODE`）——已实测，两者无差别

默认 `additive`：Ray stage 的服务 + 序列化 + 往返全部记在 worker 的关键路径上
（历史语义，也是"max(ray+local, smp)"的由来）。

`CEDAR_DP_RAY_MODE=lane`：worker 只付序列化 + 往返，stage 服务记在 Ray lane 上，
最终得分 `max(local, ray, smp, gpu)`——远端机器因此可以**增加容量**。

**实测结论（`tmp_analysis/dp_lane_probe.sh`，同一 profile、同一预算）**：

| 负载 | additive | lane | 选中计划 |
|---|---|---|---|
| simclr | 476.1 rec/s | 475.0 rec/s | 相同（W=8 + SMP 流水线，无 Ray） |
| simclrv2 | 477.1 rec/s | 464.9 rec/s | 相同（W=8 + SMP 流水线，无 Ray） |

也就是说：在这两个负载上 **DP 根本不会选择 Ray stage**（它的隔离代价 + 传输
代价始终高于本地/SMP），所以 `max(ray+local, smp)` 与 `max(local, ray, smp)`
给出同一个计划、同一个实测吞吐。这个实验可以回答之前那个设计问题：在这批
负载上两种写法的差异是零——真正决定结果的是"选不选 Ray"，而不是"Ray 记在哪条
lane"。开关保留（默认 additive，保持历史语义），供以后出现"计划确实含 Ray"的
负载时再测。

## 4.5 段长上限的教训

先在 chain-partition 搜索里加了"融合段不超过 6 个算子"的上限来换速度，结果
simclrv2_views4 从 489.8 掉到 169.5 rec/s——**实测最优计划里那条融合块正好有 7
个算子**，被上限切掉了。现在默认不设上限（`CEDAR_DP_CHAIN_MAX_SEGMENT=0`），
速度靠惰性表 + 可调 beam 来保证。

## 4.6 SMP 记账方式（`CEDAR_DP_SMP_MODE`）——实测差异很大

代码里 SMP 有两种语义：`lane`（每个 SMP stage 自成一条并行通道，harness 目前
默认）与 `additive`（SMP 服务落在 worker 的关键路径上，代码默认）。在
simclrv2_cache 上同一 profile、同一预算：

| 模式 | PICO | Plumber | 比值 | simple-DP | 对 ablation |
|---|---|---|---|---|---|
| lane | 302.5 | 262.9 | 1.15× | 442.3 | **0.68×** |
| additive | **423.1** | 262.0 | **1.61×** | 398.9 | **1.06×** |

四个主负载上的两种模式对照（同 profile、同预算）：

| 负载 | additive PICO / ablation | lane PICO / ablation |
|---|---|---|
| simclr | **490.2** / 1.17× | 475.9 / 1.09× |
| simclrv2 | **492.6** / 1.08× | 480.0 / 1.12× |
| simclrv2_views4 | 464.2 / 1.18× | **472.7** / 1.10× |
| simclrv2_cache | **423.1** / **1.06×** | 302.5 / **0.68×** |

additive 在 3/4 个负载上更好（cache 上高 40%，并把它从"PICO 比 ablation 慢 32%"
变成"快 6%"），所以筛查协议默认值已改为 `additive`（也是代码默认）。

也就是说：把 SMP 记成"免费并行通道"会让 DP 选出一个实测慢 28% 的计划。这正是
用户最初质疑 `max(ray+local, smp)` 的同一个问题在 SMP 侧的体现。四个主负载上
两种模式的对照实验正在跑（`/tmp/screen_smpmode2`），结果会决定协议默认值。

## 4.7 仍然存在的问题（诚实记录）

- **dino（2 视图）退化**：affine profile + 无竞争曲线时 PICO 选了 W=16（212.2
  rec/s），比之前 W=32 的 279.9 慢，也低于 Plumber 244.0。说明"worker 数 vs 每
  worker 宽度"的取舍在这条 recipe 上仍未被模型算对。
- **blip/clip/alpaca/pile_pubmed 的 profile 里带着从 alpaca 复制的占位竞争曲线**
  （已在本轮删除，避免占位数据驱动决策）。
- dino_views4 1.09×、swav_views4 1.06×：两条 4 视图 recipe 都是 3–8 MB/记录，
  机器带宽/算力饱和，Plumber 的朴素全本地计划已到极限。

## 5. 实验结果（本轮）

| 负载 | PICO | 最优外部 | 比值 | simple-DP | 对 ablation | 备注 |
|---|---|---|---|---|---|---|
| coco | 263.6 | Plumber 35.8 | 7.36× | 241.9 | 1.09× | |
| simclr | 490.2 | Plumber 258.0 | 1.90× | 418.0 | 1.17× | affine 生效后 |
| simclrv2 | 492.6 | Plumber 250.9 | 1.96× | 455.2 | 1.08× | affine 生效后 |
| simclrv2_views4 | 489.8 | Plumber 272.8 | 1.80× | 456.0 | 1.07× | affine 生效后 |
| simclrv2_cache | 409.5 | Plumber 254.8 | 1.61× | 451.0 | 0.91× | 筛查协议传了 `--disable_caching`，所以这个负载实际上没有用 cache；PICO 选了"无融合的朴素链" W=32（409.5），而 simple-DP 融合成 2 块（451.0）——融合定价在这里是反例，值得单独查 |
| llava_pretrain | 15.7 | DJ-Cedar 45.0 | 0.35× | 42.5 | 0.37× | 见下 |

注意：simclr/simclrv2 重测时 profile 也一并重建，simple-DP 的数字随之变化
（418/455/456），所以"affine 的净效果"要等两种 profile 对照实验才能严格分离；
但 PICO 自身从 483/455/469 提到 490/493/490，且 5 个负载上 PICO ≥1.5× 最优外部
系统（coco 7.36×、simclrv2 1.96×、simclr 1.90×、simclrv2_views4 1.80×、
simclrv2_cache 1.61×）。

llava 仍然 0.35×：说明"宽度 credit"还不够——真实计划里 8 个 actor 各带一份
CLIP+BLIP 权重、按 500 条一批提交，而 profile 的隔离测量是 bs=10 的单算子回放。
下一步要么把 CUDA 算子的**提交批大小**也测成曲线（bs=1/10/100），要么直接测
"整条链融成一个 Ray stage"的 stage 级代价。
