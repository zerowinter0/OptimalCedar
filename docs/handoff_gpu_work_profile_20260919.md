# 交接：LLaVA CLIP/BLIP 工作量 profile 与 DP 成本模型（2026-09-19）

## 1. 用户目标与范围

本轮用户说“按你说的做”，接续此前对 LLaVA 三种物化计划的分析。用户要解决的核心问题是：现有优化器把 CLIP/BLIP 的 GPU 计算成本随输入记录的**序列化字节数**线性放大，而 CPU 预处理主要增加不被这两个模型读取的元数据。正式论文实验需要一个有测量证据、可复现、不会破坏旧 optimizer profile 的修正。不能为了得到想要的排名放宽置信门槛、剔除不利样本、调小正式负载等。

本轮工作仅针对 profile 与成本模型及这三个现有 LLaVA 固定计划的估价验证；没有收到重新运行六负载完整 optimizer 矩阵的指令，也没有新跑三种计划的 end-to-end 执行。后续 agent 应先把当前实现和证据做完整，再决定是否需要新的正式运行。历史六负载为 Simclrv2、Simclrv2-cache、Commonvoice、COCO、LLaVA pretrain、StackExchange；当前协议要求非 cache 关闭 cache、cache 负载开启对应优化，所有 Ray actor（profile 与执行）必须远端。旧 optimizer 读旧 Cedar profile 条目，simple-DP/PICO 读附加的新层次 profile 条目，同一份共享 profile 要同时兼容两类 optimizer。

## 2. 环境、工作目录和不可忽视的约束

- 宿主仓库：`/home/xieruiyang/OptimalCedar`；本机实际代码 Docker：`optimalcedar-torch201-dev`，容器内 `/workspace/OptimalCedar`。**运行或修改任何代码之前必须进入实际 Docker 并 `source env/bin/activate`。**
- 远端 Ray GCS：`172.23.166.105:6379`；远端 SSH：`xieruiyang@172.23.166.105`；远端 Docker：`optimalcedar-ray-remote`。Ray 驱动节点 `172.23.166.103` 有一块 GPU 但没有 `cedar_remote` 资源；远端节点 `172.23.166.105` 有一块 GPU 和 `cedar_remote=1`。需要 `CEDAR_RAY_REQUIRE_REMOTE=1`、`CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote`，自定义脚本在构造数据集前调用 `configure_remote_ray_experiment()`。
- 长实验按 AGENTS.md 用 `nohup`、日志离线运行。勿删除或覆写已有实验数据。仓库已有大量本轮之前的未提交修改和 `outputs/` 目录；不要把 `git diff` 的全部内容归因为当前任务，也不要 wholesale reset。
- 远端代码不会自动和本机同步。2026-09-19 本轮已把 `cedar/service/ray_service.py` 同步到远端容器；本机与远端 SHA256 均为 `ec72e3d93d03cf14b67f5da84560bdde99ad4619dc7c5436795dac6eab4fa260`。如果再次修改远端 actor 依赖的代码，务必重新同步并验证。
- 论文级要求：如果测量无法证明零斜率，结果须标为 unresolved，不要强制选择 per-record；传输成本仍要按真正的总传输字节量计价。

## 3. 原始现象与已有真实执行结果

固定计划实验档案：`outputs/llava_three_fixed_plans_20000_20260919/`，正式执行同一批 20,000 输入，三个计划都有 17,837 输出和相同的输出 ID checksum。结果：

| 计划 | wall time | 输入吞吐 | 中段输出吞吐 |
| --- | ---: | ---: | ---: |
| 所有算子融合在 1 个 Ray actor，W=1 | 1177.56 s | 16.98/s | 15.50/s |
| CPU 算子 Ray width=63，GPU 算子 Ray width=1 | 1090.41 s | 18.34/s | 16.73/s |
| CPU 算子 SMP width=63，GPU 算子 Ray width=1 | 1003.76 s | 19.93/s | 18.05/s |

原成本模型对这三个计划的精确 replay 估价（ms/输入）为 32.0968、53.7880、54.0802，错误地强烈偏向全融合。旧模型把 BLIP+CLIP 的费用从当前顺序下的约 28.72 ms 加到“所有 CPU 在 GPU 前”时的 53.57 ms，原因是预测 GPU 输入序列化大小由约 999 B 增到约 1863 B，却把这一字节增长直接当成 GPU 计算量增长。旧 replay 和可运行脚本在 `outputs/llava_forced_cpu_width_cost_20260919/`。

旧共享 profile `outputs/simple_dp_three_transport_all_20260918/llava_pretrain/profiles/shared.yaml` 的 `physical_model.operator_affine` 没有 GPU pipe 1/2 的斜率条目；旧 `operator_compute_scaling` 对这两个 pipe 记录 `insufficient_size_variation`、confidence 0、默认 per_data。两个 GPU 算子是 `evaluation/pipelines/llava_pretrain/dj_operators.py` 中的 `ImageTextMatchingFilter`（pipe 1，BLIP）和 `ImageTextSimilarityFilter`（pipe 2，CLIP）。其真正模型调用读取 `raw_content`、`images` 和 `context.image_root`，不会读取任意新增顶层元数据。是否真的调用模型还取决于有图片路径并且文本存在 `<image>` chunk；只有这类有效记录才能用于 GPU 工作量测量。

## 4. 当前实现及主要文件

当前工作树已加入但**尚未完成验收**：

- `cedar/client/cuda_work_profiler.py`：对 LLaVA CLIP/BLIP，从 predecessor 的合法 pickle 快照中过滤实际会调用模型的记录；给同样的图文内容添加 2048 B 顶层未用元数据；做 original/padded/padded/original (ABBA) 远端 actor 计时。判定要求至少 1.5× 序列化字节对比、每种条件至少两次收敛计时，`(|均值差| + max(2×最大标准误，四次均值极差))/原始均值 <= 10%`；还要求四次 trial 同一块远端物理 GPU。通过时写 `accepted=true`，否则明确原因。该测试只能证明“额外未用元数据”的斜率，不证明图片数量、图像内容或 token 长度变化的斜率。
- `cedar/client/dataset.py`：默认 dual profile 把 CUDA work 证据附到 `physical_model.cuda_workload`，只对 accepted GPU pipe 写 `operator_compute_scaling=per_record`；旧 baseline/offload/TF 条目仍保留。TF 旧候选已被移到隔离 profile 之前测，避免新测量给旧候选增加额外 cache/model warming。`_adaptive_operator_benchmark` 新增 `record_actor_locations`，以及**刚开始写但还没接入 CUDA profiler 的** `snapshot_sequence` 参数，用于在一个物化 actor 内按顺序测多组快照。
- `cedar/service/ray_service.py`：RayActor 新增 `get_runtime_location()` 返回节点 IP/GPU ID。远端容器已同步该文件，否则 Ray actor import 会失败。
- `cedar/compose/simple_dp_ablation_optimizer.py`：新的层次 profile simple-DP 对 `per_record` 用存活记录数递推、按 baseline reach 归一化，accepted GPU 不再留一个与 mask 无关的 affine 固定费用；边界按进入 stage 的请求数计固定开销，按过滤后输入和输出真实总字节数计传输；workers/width 变体的共享传输部分同步改为按存活总字节数。旧 `OldDpBoundaryOptimizer` 保持旧 Cedar 路径。
- `cedar/compose/my_optimizer.py` 和 `cedar/compose/affine_dp_cost.py`：PICO/DP 消费 accepted CUDA 证据，去掉 metadata byte slope，但保留测得的**绝对计算 anchor**；先前一个版本错误地把 affine anchor 直接改成 1 ms，现已用测试修正。
- `tests/test_cuda_work_profile.py`：相关单测；另有 `tests/test_simple_dp_ablations.py` 的已有 isolated stubs 补了 cardinality，`tests/test_dp_cache_fusion_optimizer.py` 中一个本来就与注释矛盾的“DpOptimizer 必须选 Ray”断言改为验证本地计划有效。`README.md` 已描述新 profile 契约和 CUDA 证据限制。

## 5. 实验记录：当前不能声称“GPU 零斜率已被最终证明”

所有实验都保存在各自目录，失败日志不要删除：

1. `outputs/llava_cuda_work_calibration_20260919/`：第一版 ABBA，**未筛掉 GPU no-op 输入、未记录 actor 物理位置**。BLIP/CLIP 均 accepted，均值约 32.74→33.57 和 20.86→20.99 ms。其 `RESULT.md` 与 `comparison.json` 反映的是当时尚未修正 fixed fraction/总传输字节等问题的代码，**只可作历史诊断，不可引用为最终结果**。
2. `outputs/llava_cuda_work_calibration_active_20260919/`：筛掉 no-op 输入，仍是每 trial 独立 actor。BLIP accepted，CLIP 因 actor 间波动 unresolved。
3. `outputs/llava_cuda_work_calibration_placed_20260919/`：开始记录位置，但远端容器未同步新 `RayActor.get_runtime_location`，actor import 报 `AttributeError`。失败日志留存。之后远端文件已经同步。
4. `outputs/llava_cuda_work_calibration_verified_20260919/`：32 个真正调用模型的合法输入/算子，四个 trial 都是远端 `172.23.166.105` GPU `0`，计时各自收敛；然而独立 actor 的离群波动使 **BLIP 和 CLIP 都 rejected**。BLIP 原始均值 33.041 ms、加元数据 34.283 ms、保守上界变化 14.6%；CLIP 原始 20.872 ms、加元数据 23.392 ms、上界变化 36.6%。原始字节对比仍分别约 4.24×、4.43×。详见 `cuda_work.json` 和 `calibrate.log`。该目录的 `shared.yaml` 没有为 GPU 两管道加入 accepted per_record 注释。此版原本预置了 `score.py` 和 `report.py`，**因为证据不成立，目前没有运行最终 score/report**。不能把旧的 32.10→57.44 等 replay 数字当作当前最终模型结果。

全部四次校准已结束；交接时没有 `calibrate.py` 或 `score.py` 在后台运行。

## 6. 目前停下来的具体位置与已知失败

为消除跨 actor 启动、时钟、cache 状态的波动，正把 ABBA 四组快照改为**同一个已经预热的 Ray actor/variant 内连续测量**，每个阶段切换 replay 快照、重新 warmup、重置 worker 计时统计，再测到置信条件。这样仍需位置记录，但四个阶段天然同 actor、同 GPU。

`cedar/client/dataset.py::_adaptive_operator_benchmark` 目前已经加入 `snapshot_sequence: Optional[List[Tuple[str,List[bytes]]]]`，若提供则循环四个 trial 并只构造/关闭一次 variant；默认单 trial 返回原 dict，以兼容旧调用。`cedar/client/cuda_work_profiler.py` **还没有改成传 `snapshot_sequence`，仍调用 benchmark 四次创建四个 actor**。这是最关键的未完成改动。

新回归测试 `tests/test_cuda_work_profile.py::test_adaptive_series_reuses_one_warmed_actor_for_counterfactuals` 已写但当前失败：测试中的 fake pipe 是 `SimpleNamespace`，缺少 `execution_resource`，在 `_ray_profile_gpu_fraction` 先抛 `AttributeError: 'types.SimpleNamespace' object has no attribute 'execution_resource'`。下一步先给 fake pipe 加 `execution_resource=PipeExecutionResource.CPU`，若随后暴露别的 fake 协议缺口继续修正，使测试真正断言“只创建一次 actor、各阶段返回不同的 worker mean”。不能为了让测试通过而弱化真实断言。

在这项尚未接入的 series 修改之前，相关测试最后一次完整运行是 `78 passed, 5 warnings in 1.56s`（`outputs/llava_cuda_work_calibration_placed_20260919/tests.log`），`py_compile` 和 `git diff --check` 通过。此后新增的 series 单测失败，**不可再声称当前全部测试通过**。请在完成 series 后重新运行完整相关集。

## 7. 建议接续步骤（按依赖顺序）

1. 在本机实际 Docker 内激活 env，修正上述 fake pipe 单测并确保它由红转绿；检查 `snapshot_sequence` 的运行路径对原有单 trial 不改变测量语义和返回格式。相关命令：`pytest -q tests/test_cuda_work_profile.py -k adaptive_series_reuses_one_warmed_actor`。
2. 把 `profile_cuda_metadata_invariance` 的四个独立 `_adaptive_operator_benchmark` 调用改为**一次**调用，传 `snapshot_sequence=[('original', original), ('padded', padded), ('padded', padded), ('original', original)]` 和 `record_actor_locations=True`。把返回四个 dict 按标签写进原 `runs` schema；修改 `method`、README 和实验报告文字为“同一 actor 内 ABBA”。同一 actor 的四个位置应该一致；保留 `assess_actor_placement` 验证，防止未来代码退化。不要沿用声称“独立 actor”的旧 method 字段。
3. 运行完整定向测试：`pytest -q tests/test_cuda_work_profile.py tests/test_simple_dp_ablations.py tests/test_affine_dp_cost.py tests/test_layered_affine_profile.py tests/test_dp_cache_fusion_optimizer.py tests/test_ray_actor_options.py`；再 `python -m py_compile` 相关改动文件和 `git diff --check`。
4. 用 `outputs/llava_cuda_work_calibration_verified_20260919/calibrate.py` 为模板创建**新的**离线实验目录，修改脚本内 `out` 路径，以 `nohup` 运行。仍用旧共享 profile 的合法字段作基础、重新捕获 300 条样本的 reservoir、每算子取最多 32 个确实会调用模型的输入，`CEDAR_RAY_REQUIRE_REMOTE=1`、`CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote`、HF 离线。当前 remote `ray_service.py` 已同步；如又改了远端 actor 类，需重新同步。不要覆写 1–4 轮原始日志。结束后查看每个 trial 的 `adaptive_profile.converged`、位置、原始 mean/stderr、accepted/reason。
5. 如果同 actor 仍未通过 10% 等价门槛，先诊断 GPU 干扰或计时协议，必要时做额外预注册的重复/分层测量；**不要**事后删除离群点、降低测试门槛、沿用最早一轮 accepted 结论。若证据最终仍不足，保持 `accepted=false`，把本轮研究结果如实报告为“当前不能识别元数据斜率”。图片数量/token 长度斜率仍需要独立受控干预才能声称测出来。
6. 只有在有合格 accepted profile 后，使用当前代码对原三种固定物化计划做 exact objective replay（复制 `outputs/llava_cuda_work_calibration_verified_20260919/score.py`，修改 `out` 指向新的目录），与已经完成的 20k 真实执行比较。再基于新数据运行/生成报告。若出现 Ray/SMP split 排序残差，把它归于 CPU stage/transport 模型继续调查，不要笼统声称“整个 cost model 已准确”。
7. 验证新共享 profile 中 `baseline`、`offloads`、`disk_info`、TF 等旧条目与输入基础 profile 一致；accepted 注释只增加新字段。注意本次针对 LLaVA 是**补充校准、复用旧 profile 条目**；未来正式矩阵要从头生成兼容全部 optimizer 的完整 dual profile。

## 8. 可直接使用的关键命令与路径

进入本机容器执行测试：

```bash
docker exec optimalcedar-torch201-dev bash -lc 'source env/bin/activate && pytest -q tests/test_cuda_work_profile.py tests/test_simple_dp_ablations.py tests/test_affine_dp_cost.py tests/test_layered_affine_profile.py tests/test_dp_cache_fusion_optimizer.py tests/test_ray_actor_options.py'
```

远端文件同步示例（只有改动了远端 actor 依赖代码才需要；先比较差异）：

```bash
ssh xieruiyang@172.23.166.105 "docker exec -i optimalcedar-ray-remote bash -lc 'source env/bin/activate && cat > cedar/service/ray_service.py && python -m py_compile cedar/service/ray_service.py'" < cedar/service/ray_service.py
```

当前重要文件：
- `cedar/client/cuda_work_profiler.py`
- `cedar/client/dataset.py`（`_profile_standard`、`_adaptive_operator_benchmark`、`_profile_layered_backends`）
- `cedar/service/ray_service.py`
- `cedar/compose/simple_dp_ablation_optimizer.py`
- `cedar/compose/my_optimizer.py`
- `cedar/compose/affine_dp_cost.py`
- `tests/test_cuda_work_profile.py`
- `outputs/llava_three_fixed_plans_20000_20260919/`
- `outputs/llava_forced_cpu_width_cost_20260919/`
- `outputs/llava_cuda_work_calibration_verified_20260919/`

交接时未提交代码、未推送、未完成新同 actor 校准与最终 replay。请先从上述失败测试和未接入的 series 调用继续，勿把本文件中的历史尝试当作最终论文结论。
