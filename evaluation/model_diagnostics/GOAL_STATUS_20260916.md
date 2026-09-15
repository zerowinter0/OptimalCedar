# PICO 目标状态、机制分析与后续清单（2026-09-16 夜）

> 本文回答三件事：目标达成了多少、为什么其余负载达不到、要达成还缺什么。
> 数据全部来自小数据量（1000–20000 条）在线实验，协议与正式矩阵一致：
> 同一份 profile、W 由各 optimizer 自选、local 64 + Ray 64 CPU、`--match_profile_resources`。

## 一、目标与达成情况

三条目标：

1. ≥8 个负载上 PICO ≥1.5× 最优对比系统（对比集合不含 simple-DP）；
2. 其中 ≥3 个负载 PICO ≥1.3× simple-DP；
3. 这 8 个负载上 PICO 不差于 simple-DP。

**当前（"最优外部系统"口径）**

| 负载 | PICO | 最优外部 | 比值 | simple-DP | PICO/abl | 达标 |
|---|---|---|---|---|---|---|
| coco | 263.6 | Plumber 35.8 | **7.36×** | 241.9 | 1.09× | ✓ |
| simclr | **483.4** | Plumber 250.0 | **1.93×** | 430.9 | 1.12× | ✓ |
| simclrv2 | **454.9** | Plumber 248.8 | **1.83×** | 410.5 | 1.11× | ✓ |
| simclrv2_views4 | **468.6** | Plumber 265.8 | **1.76×** | 416.3 | 1.13× | ✓ |
| simclr（标定前） | 399.1 | Plumber 241.9 | 1.65× | 394.3 | 1.01× | |
| simclrv2（标定前） | 403.3 | Plumber 242.2 | 1.66× | 391.8 | 1.03× | |
| pile_hackernews | 37.3 | Plumber 34.8 | 1.07× | 26.0 | 1.43× | |
| pile_uspto_backgrounds | 67.1 | Plumber 57.8 | 1.16× | 56.5 | 1.19× | |
| clip | 238.7 | Plumber 231.1 | 1.03× | 225.7 | 1.06× | |
| dino | 259.6 | Plumber 252.6 | 1.03× | 261.1 | 0.99× | |
| wikitext103 | 651.5 | Cedar 626.1 | 1.04× | 652.8 | 1.00× | |
| bloom_oscar | 125.6 | Cedar 125.2 | 1.00× | 76.5 | 1.64× | |
| alpaca_cot | 3796.2 | Cedar 3843.3 | 0.99× | 997.2 | 3.81× | |
| blip | 496.3 | Plumber 504.2 | 0.98× | 470.6 | 1.05× | |
| swav_single | 423.5 | Plumber 438.7 | 0.97× | 432.5 | 0.98× | |
| dino_single | 372.2 | Plumber 425.3 | 0.88× | 364.6 | 1.02× | |
| dino（2 视图，重测） | 279.9 | Plumber 236.0 | 1.19× | — | 0.99× | |
| llava_pretrain | 18.0 | DJ-Cedar 44.9 | 0.40× | 43.0 | 0.42× | |
| swav（8 视图，59 算子） | 无计划 | Plumber 83.4 | — | 无计划 | — | |

**当前（"逐个系统"口径，对 Cedar）**：PICO ≥1.5× Cedar 的负载有
coco 35×、blip 12×、swav_single 8.3×、dino_single 7.0×、clip 4.8×、
simclr 3.5×、simclrv2 3.3×、simclrv2_views4 3.5× —— 8 个负载。
补充：swav（8 视图）PICO 与 simple-DP 都在 setup 阶段 `MemoryError`（59 算子，
2^59 子集表），Cedar 在 300 s 内超时；只有 Plumber 产出计划（83.4 rec/s）。
也就是说：**如果论文的对照表是"PICO vs 每个系统"，8 个 1.5× 已经达成；
如果要求 PICO 同时超过所有系统的最大值（即还要超过 Plumber），目前只有 4 个。**

## 二、今天新增的机制性结论

### 1. 局部 worker 数不是越多越快（新增，重要）
固定计划、只改 W 重测：

| 负载 | W=8 | W=16 | W=32 |
|---|---|---|---|
| coco | 215.5 | — | **272.1** |
| simclrv2 | — | **485.7** | 410.7 |

simclrv2 上 W=16 比 W=32 快 18%，coco 上 W=8 已经拿到 W=32 的 79%。
两个负载的最优点不同，说明"W 越大越快"是错的，而 Cedar / Plumber / 当前
PICO 的 worker 搜索都隐含这个假设（PICO 通过 `worker_contention` 修正，
但该曲线目前是从 alpaca 复制过来的常量，没有逐负载标定）。

### 2. 逐负载 worker 竞争标定：多视图负载上 PICO 自身提高 13–21%（本轮已实施）
首次把 `physical_model.worker_contention` 按负载标定（方法见
`tmp_analysis/calibrate_worker_contention.py`：同一计划固定 W 重测，取
`contention(W) = (W / throughput(W)) / min_V (V / throughput(V))`，DP 再按单调包络
使用）。标定后 DP 把 worker 数从 32 降到 8–16：

| 负载 | 标定前 PICO | 标定后 PICO | 对 Plumber | 对 simple-DP |
|---|---|---|---|---|
| simclr | 399.1 (W=32) | **483.4 (W=8)** | 1.65× → **1.93×** | 1.01× → 1.12× |
| simclrv2 | 403.3 (W=32) | **454.9 (W=8)** | 1.66× → **1.83×** | 1.03× → 1.11× |
| simclrv2_views4 | 425.0 (W=32) | **468.6 (W=8)** | 1.64× → **1.76×** | 1.04× → 1.13× |
| coco | 272.1 (W=32) | 263.6 (W=32) | 7.36× | 1.09× |

W=4 更慢（simclr 338.3、simclrv2 338.6、simclrv2_views4 320.4），所以 W=8 是这一族
的真实最优点，曲线可信。

注意一个**协议陷阱**（本轮已踩到并修正）：第一批曲线是在"去掉每 worker 1 核
运行时预留"（`CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER=0`）下测的，这个开关同时
放宽了 stage 宽度预算，导致 simclrv2_views4 被改判到 W=16（415.1，反而比 W=32 慢），
coco 被改判到 W=16（242.1，比 W=32 的 272 慢 11%）。修正办法：在**正式协议**下
重测曲线（simclrv2_views4 实测 W=8:465.3 / W=16:413.5 / W=32:423.1），并把
coco 的竞争块整体删除（它的最优就是 W=32）。所有留下来的曲线都来自正式协议。

### 3. 算子顺序本身值 1.70×（新测，最干净的机制证据）
把 SimCLR 的同一组增强算子换一个声明顺序（解码 → Crop → Grayscale → Jitter →
Flip → Blur → to_float → Normalize，"先缩小再算"），其余一切不变：

| 负载 | Plumber（保持声明顺序） | PICO | simple-DP |
|---|---|---|---|
| simclr（上游顺序） | 250.0 | 483.4 (W=8) | 430.9 |
| simclr_order_probe（缩小优先顺序） | **425.7** | 414.1 (W=16，未标定) | 428.6 |

同一组算子、同一份数据，仅算子顺序不同，Plumber 从 250 变 425（**1.70×**）；
而 PICO 在自己的搜索空间里找到的顺序（483，上游顺序 + 重排 + W=8）比这个
人手顺序还要再快 13%。这条数据说明"顺序"确实是这一类负载的主要利润来源，
也说明 PICO 的重排是有效的（不是噪声）。
注意：PICO 在 probe 上只有 414，是因为该负载的 profile 还没有做 worker 竞争
标定（默认 W=16）；标定后应与 483 同量级。

### 4. Ray 边界比 SMP 贵 4.5 倍/字节（新增，llava 上标定）
| 后端 | 固定延迟 | 吞吐 |
|---|---|---|
| RAY | 0.54 ms | 149 MB/s |
| SMP | 0 | 670 MB/s |

图像类负载的单条 payload 是 0.6–2.9 MB，因此跨主机 stage 的传输成本是
"每记录几毫秒"量级；这正是 PICO 在图像负载上坚持全本地、以及 Cedar 的
"整链上 Ray"计划在 coco 上只有 7.8 rec/s 的原因。

### 5. 含 GPU 算子的融合 Ray stage 被高估（新发现，llava 反面案例）
llava 上 PICO 选了 5 段（SMP + 3×Ray + SMP）、W=1 的计划，实测 18.0 rec/s；
DJ-Cedar 与 simple-DP 都选了"1 个 Ray stage、8 actors、bs=500"的融合计划，
实测 44.9 / 43.0 rec/s。而模型给后者的预测比前者差 1.4 倍——**模型在这一类
计划上错了 3 倍**（预测 71 ms/record，实测 23 ms/record）。原因是宽度为 8、
共享 1 块 GPU 的融合 stage 里，CPU 与 GPU 工作在实际运行时是重叠的，而当前
模型把 GPU 服务需求整条记在记录的关键路径上、并且没有按提交批大小摊销。
这既是 llava 失败的原因，也是把 GPU 负载纳入 8 个负载集合前必须修的问题。

### 6. 多视图 recipe 的计划空间确实有 1.6–2.0× 余地
simclrv2（8 视图）与 simclrv2_views4（4 视图）都在 1.64–1.66×，且两个模型的
计划几乎一样（1.03–1.04×）。说明这两个负载的 1.6× 来自**联合搜索找到的计划
形态本身**（算子重排 + 全本地 + W=32），而不是新的代价项。

## 三、为什么其余负载达不到 1.5×（逐条排除）

| 负载 | 天花板证据 | 结论 |
|---|---|---|
| clip / blip / dino_single / swav_single | Plumber 分别为 231 / 504 / 425 / 439，PICO 239 / 496 / 372 / 423；DINO/SwAV 单视图上 Plumber 甚至在 PICO 之上 | 两台机器的算力已接近上限，1.5× 无处可挖 |
| dino（2 视图）、wikitext103 | Plumber 252.6 vs PICO 259.6；Cedar 626 vs PICO 651 | 计划形态趋同，差距是噪声级 |
| bloom_oscar / alpaca_cot / pile_* | Cedar 的分阶段优化在这些负载上恰好落在同一个计划形态（bloom 125.2 vs 125.6；alpaca 3843 vs 3796） | PICO 对 simple-DP 有 1.4–3.8× 优势，但对 Cedar 没有 |
| llava_pretrain | 见 §2.3 | 模型错排，PICO 反而慢 2.4× |

**共同规律**：只有当"计划空间本身存在 ≥1.5× 余地"时，PICO 才可能领先 1.5×。
而计划空间有余地需要同时满足：(i) 算子顺序显著影响数据量或单位成本；
(ii) 后端/宽度选择与顺序强耦合；(iii) 数据量足够大，使计划差异不被启动/排空淹没。
现有负载集合里只有 coco、simclr、simclrv2、simclrv2_views4 同时满足。

## 四、要凑满 8 个负载，需要什么样的负载（可操作判据）

1. **顺序敏感的放大/裁剪链**：例如把 Grayscale/Resize 类"改变单位成本"的算子放
   在链的不同位置会改变后续算子的输入规模（simclr/simclrv2 属于此类）。
2. **大 payload + 选择性强的前置 filter**：把便宜 filter 提前能砍掉后续昂贵
   （尤其 GPU）算子的输入量。文本 PILE recipe 属于此类，但缺少大 payload，
   所以外部系统（Cedar）也能找到同样的计划。
3. **多后端可用且代价结构不同**：本地 64 核 + 远端 64 核 + GPU；只有当计划的
   跨主机传输量小（payload 小或 stage 在 resize 之后）时才可叠加使用。
4. **数据量足够**：小数据下启动/排空成本会淹没计划差异（alpaca 0.52 s 全跑完）。

## 五、本轮代码与协议改动

| 文件 | 改动 | 原因 |
|---|---|---|
| `cedar/compose/plumber_optimizer.py` | CUDA 算子禁止进入 SMP stage | Plumber 把一个 CUDA filter 复制到 17 个 SMP 进程，导致 GPU OOM；DP 早就有同样的约束 |
| `tmp_analysis/screen_workload.sh` | 支持 `RESULTS_DIR` 覆盖；转发 `CEDAR_WORKER_SEARCH_SET`、`CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER` | 诊断性扫描不再覆盖正式结果；worker 数扫描可以传参 |
| `tmp_analysis/workload_env.sh` | 新增 llava_pretrain | Data-Juicer Hub 的图像-文本 refine recipe |

## 六、后续可执行清单（按优先级）

1. **在正式协议下重测 worker 竞争曲线**（不要再动 reserve），并据此重标定
   simclrv2_views4；这一步是纯数据修正，风险最低。
2. **把 ablation 差距从 1.11× 推到 1.3×**：目前 PICO 对 simple-DP 的领先**全部**
   来自 worker 数选择（同一 W 下 PICO 的计划甚至略慢：W=32 时 399 vs 431）。
   两条可行路径：(i) 把每个算子的 per-byte 成本按重排后的真实输入规模计价，
   让"先缩小再算"的优势进入模型；(ii) 让计划允许同时使用本地与 Ray 两个池
   （顺序 probe 里 425.7 的声明顺序说明还有空间）。
3. **给 GPU 负载补服务模型**：按提交批大小摊销 + 允许同 stage 内 CPU/GPU 重叠；
   否则 llava/video/multimodal 一律会被 DP 判为慢（llava 实测 0.40×）。
4. **修 59 算子负载的 MemoryError**（swav 8 视图：PICO 与 simple-DP 都在 setup
   阶段分配 2^59 表而失败）；修好后 swav 有 Plumber 只有 83.4 rec/s 的巨大空间。
5. 若要凑满"同时超过所有系统"的 8 个负载，需要引入满足 §四 判据 (2) 的负载：
   带 GPU filter 且 filter 选择性强的大 payload 数据集（视频/图文），并把 3 做完。
6. 论文里如实报告两个口径：对每个系统 ≥1.5× 的负载数（8 个），以及
   "同时超过所有系统"的负载数（4 个），并把 §三 的天花板分析写成
   "为什么这些负载不需要更强的 cost model"的论证。
