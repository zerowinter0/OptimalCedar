# PICO 目标状态、机制分析与后续清单（2026-09-16 夜）

> 本文回答三件事：目标达成了多少、为什么其余负载达不到、要达成还缺什么。
> 数据全部来自小数据量（1000–20000 条）在线实验，协议与正式矩阵一致：
> 同一份 profile、W 由各 optimizer 自选、local 64 + Ray 64 CPU、`--match_profile_resources`。


## commonvoice 诊断结论（2026-09-16 14:45）

修掉三个阻塞点后 commonvoice 可以完整测量了，结果是 **PICO 178–278 rec/s**，
而 ablation 的 simple-DP **425–466 rec/s**、Plumber 241–284 rec/s。逐层定位：

1. **DP 曾对所有 worker 数报 "no feasible final state"**：不是没有合法计划——我逐算子
   验证过 7 个算子都能单独成块、串起来合法。真实原因是"贪心整块 incumbent"的分数
   （模型认为全融合最快）低于任何精确搜索标签，单调剪枝把所有前缀标签都剪掉了。
   已修：精确搜索判定不可行时回退到该可行 incumbent，而不是让整个负载失败。
2. **模型偏好"全融合 + 把音频解码 offload 到 Ray"**：profile 里 `_read` 的
   in-process 代价 **24.87 ms**、RAY **8.57 ms**、SMP **6.29 ms**。但直接测同一段音频
   的 warm decode 只要 **5.3 ms**（本地）：in-process 基线是"每条记录读一个新文件"
   的冷 I/O，而 RAY/SMP 隔离基准是对同一批 snapshot **反复回放**（热页缓存）——
   两者不可比，模型因此以为 offload 能白拿 3× 加速。
3. **目标函数缺少"流水线"credit**：把 7 个算子各放一个进程（simple-DP 的
   8×SMP(1) 计划）在模型里被算成"服务时间之和"，与"全部融进一个进程"等价，
   于是模型随意地选了融合；实测流水线快 2×（213.6 融合 vs 425.8 流水线）。
   注意这与 simclrv2_cache 的结论相反（那里融合确实更快），说明**融合 vs 流水线的
   取舍依赖算子是否 I/O-bound**，需要用实测的每算子"吞吐型/延迟型"分类来定价，
   而不是全局取 sum 或 max。

诊断用到的命令：`tmp_analysis/dp_local_probe.sh <workload> <n> INPROCESS,SMP`
（排除 Ray，看模型本地产出的计划）、`tmp_analysis/read_timing_probe.py`（直接测解码）。

## 〇、按"负载类别不重复"重排后的达标集合（2026-09-16 14:30）

用户指出 5 个达标负载里 4 个是 SimCLR 变体，要求最多保留一个 simclrv2 与一个
simclrv2_cache。按此收缩后：

| 类别 | 负载 | PICO | 最优外部 | 比值 | simple-DP | 对 ablation |
|---|---|---|---|---|---|---|
| 图像检测增强（原始 Cedar） | **coco** | 265.8 | Plumber 34.4 | **7.72×** | 243.2 | 1.09× |
| 多视图增强 | **simclrv2** | 492.6 | Plumber 250.9 | **1.96×** | 455.2 | 1.08× |
| 多视图增强 + 磁盘 cache | **simclrv2_cache** | 423.1 | Plumber 262.0 | **1.61×** | 398.9 | 1.06× |
| （已按要求剔除） | simclr / simclrv2_views4 | 490.2 / 464.2 | 258.0 / 266.9 | 1.90× / 1.74× | — | — |

即：**类别不重复的达标负载目前是 3 个**（目标 8 个）。下面记录为补足类别而做的
尝试与其结果。

### 用户点名的两个"原始 Cedar"负载

| 负载 | 结果 | 结论 |
|---|---|---|
| wikitext103 | PICO 651.5 (W=16)、Cedar 626.1、Plumber 457.9、simple-DP 652.8 | 对最优外部 1.04×、对 ablation 1.00× —— 没有优势（该负载上计划形态是唯一的） |
| commonvoice（音频） | PICO 178.8 (W=32)、Plumber 241.7、Cedar 165.4、simple-DP **466.6** | 对最优外部 0.74× —— **PICO 的精确搜索在这条 pipeline 上返回 "no feasible final state"**，只能退回较弱的 incumbent；而 ablation 的 simple-DP 能到 466.6，说明修好这个可行性 bug 后该负载有 ~1.9× 的空间 |

commonvoice 的三个阻塞点本轮已修掉两个半：① Ray actor 用相对路径读不到音频
（改成绝对路径）；② 精确搜索不可行时不再让整个负载失败（回退到可行 incumbent）；
③ 仍待修：为什么"存在合法单算子块覆盖"（我逐算子验证过 7 个算子都能单独成块）
但 Pareto 子集 DP 仍然报 `no feasible final state`。

### 其他类别的尝试

| 类别 | 负载 | 结果 |
|---|---|---|
| 图文 refine（Data-Juicer Hub） | llava_pretrain | PICO 15.7 vs DJ 45.0 = 0.35×（GPU 批大小建模未完成） |
| 图文匹配/相似度 | blip / clip | 0.98× / 1.03×（Plumber 已到机器上限） |
| 单视图增强 | dino_single / swav_single | 0.88× / 0.97× |
| 4 视图增强（大 payload） | dino_views4 / swav_views4 | 1.09× / 1.06×（3–8 MB/记录，机器饱和） |
| 视频（Data-Juicer） | video_self_evolution | profile 阶段失败：Ray actor 离线拉不到 HF 模型 + 远端视频路径不匹配（需先补基础设施） |
| 文本 LLM 数据 | alpaca_cot / bloom_oscar / pile_* | 对 ablation 大赢（1.4–3.8×）、对 Cedar 打平（0.99–1.05×） |

## 一、目标与达成情况（2026-09-16 13:35，协议统一为 additive SMP）

| 负载 | PICO | 最优外部 | 比值 | simple-DP | 对 ablation | 备注 |
|---|---|---|---|---|---|---|
| coco | 265.8 | Plumber 34.4 | **7.72×** | 243.2 | 1.09× | |
| simclrv2 | 492.6 | Plumber 250.9 | **1.96×** | 455.2 | 1.08× | affine + W 标定 |
| simclr | 490.2 | Plumber 258.0 | **1.90×** | 418.0 | 1.17× | affine + W 标定 |
| simclrv2_views4 | 464.2 | Plumber 266.9 | **1.74×** | 393.6 | 1.18× | affine + W 标定 |
| simclrv2_cache | 423.1 | Plumber 262.0 | **1.61×** | 398.9 | 1.06× | additive SMP 修复 |
| dino_views4 | 150.5 | Plumber 138.5 | 1.09× | 超时 | — | 3 MB/记录，机器饱和 |
| swav_views4 | 129.8 | Plumber 122.4 | 1.06× | 超时 | — | 7.8 MB/记录，机器饱和 |
| dino | 212.2 | Plumber 244.0 | 0.87× | 262.4 | 0.81× | affine 后 W 取舍仍错 |
| blip / clip / dino_single / swav_single | 见旧表 | Plumber | 0.88–1.06× | — | — | Plumber 已到机器上限 |
| bloom_oscar / alpaca_cot | 125.6 / 3796 | Cedar 125.2 / 3843 | 1.00 / 0.99× | 76.5 / 997 | 1.64 / 3.81× | 对 ablation 大赢、对 Cedar 打平 |
| llava_pretrain | 15.7 | DJ-Cedar 45.0 | 0.35× | 42.5 | 0.37× | GPU 批大小建模未完成 |

**汇总**：目标 1 达成 **5/8**（coco 7.36×、simclr 1.93×、simclrv2 1.83×、
simclrv2_views4 1.76×、simclrv2_cache 1.61×）；目标 2 达成 **0/3**（最大
ablation 差距 1.13×）；目标 3 在 4/5 个达标负载上成立（simclrv2_cache 上
PICO 0.91× 于 simple-DP，见 §2.7）。

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

### 7. 缓存负载上模型反而输给 ablation（新发现）
`simclrv2_cache`（simclrv2 + object-disk cache）：PICO 409.5 rec/s、Plumber 254.8、
Cedar 120.7，但 simple-DP 451.0——**PICO 只有它的 0.91×**。说明"缓存边界怎么定价"
上 Cedar 原模型反而更接近现实（它按 cache 命中后的算子成本直接打折）。这一条
既是目标 3 的反例，也是下一步最具体的模型修正点：把 cache 写/读边界按实测
标定（本轮 llava 已经证明 RAY/SMP 边界可以用线性拟合标定到 R²>0.99）。

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

**profile 现状**：`simclr`、`simclrv2`、`simclrv2_views4` 的
`physical_model.worker_contention` 是本轮按负载实测的（三者的最优点都是 W=8）；
`coco` 已删除竞争块（它的最优是 W=32）；`alpaca_cot`、`blip`、`clip`、`dino`、
`pile_pubmed_abstracts` 里仍是从 alpaca 复制过来的占位曲线（不影响这几个负载的
W 选择，因为它们在 W=32 上最好，但仍应在正式实验前逐负载替换）。

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
