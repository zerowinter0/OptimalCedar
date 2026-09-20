# 放大 campaign 暂停记录（2026-09-20 22:30 UTC+8）

**状态：已暂停。** 代码已修复（teardown 挂死缺陷），实验进程已停止，`outputs/ultimate_eight_optimizers_20260920` 冻结为只读快照（其 `modules/` 未被改动）。

- campaign：`outputs/ultimate_eight_optimizers_20260920`（9 optimizer × 6 负载，1 轮，单 cell 上限 2 h）
- 启动：2026-09-20 01:14 UTC+8（`scripts/run_ultimate_matrix_20260920.sh`），暂停：2026-09-20 22:30 UTC+8，累计约 21.3 h
- 完成：simclrv2 9/9、simclrv2_cache 9/9、commonvoice 9/9（`unopti` 真超时）、coco 7/9（2 个伪超时 + 1 个被打断）
- 未开始：llava_pretrain、stackexchange（输入子集已物化：50,000 / 20,000 条）

## 0. 为什么暂停

`coco` 的 `dp-boundary`（226.5 rec/s）与 `dp-boundary-affine`（240.0 rec/s）两个 cell **测量本身已跑完并落盘**，但进程在 teardown 阶段挂死，直到 2 小时上限被 runner 杀掉，因此状态被记为 `timeout`。每个 cell 白扔 2 小时，且挂死期间残留进程会污染后续 cell，所以先修代码、再暂停记录状态。

## 1. 本次修复：本地 worker 的 teardown 挂死

### 1.1 现象

`outputs/ultimate_eight_optimizers_20260920/coco/logs/round1__old_dp_boundary.log` 结尾：

```
INFO:cedar.client.dataset:Terminating worker 0...31
======Optimizer Performance Comparison======
old_dp_boundary: total=251.221272s, perf=220.751882s, samples=50000, throughput=226.499 samples/s ...
INFO:__main__:Wrote comparison results to .../round1__old_dp_boundary.json
Exception ignored in atexit callback: <function _exit_function at 0x...>
  File "/usr/lib/python3.10/multiprocessing/util.py", line 357, in _exit_function
    p.join()
  File ".../ray/_private/worker.py", line 1744, in sigterm_handler
    sys.exit(signum)
SystemExit: 15
```

结果已经写完，进程却停在解释器退出阶段（`multiprocessing` 的 atexit 对子进程做**无超时** `join`），直到 runner 超时后发 SIGTERM 才结束。

### 1.2 根因

1. 消费端在 `num_total_samples` 处停止迭代（coco：每个 feature 副本都能产出全部 50,000 条、32 个 worker 合计 1.6M 条，消费端只取走 50,000 条就停了），本地 worker 继续生产，很快阻塞在 `result_queue.put()`（`MP_QUEUE_MAX_SIZE=100`）。
2. 阻塞中的 worker 既看不到 `done` 事件（该检查只在 epoch 结束时做），也可能不响应 SIGTERM（主线程卡在不可中断的 threading 等待里）。
3. 旧的 `_shutdown()` 只有「1 s 宽限 → `terminate()` → `join(5)`」，随后直接 `clear()`；SIGTERM 无效的 worker 被静默丢弃，仍留在 `multiprocessing.process._children` 里。
4. 解释器退出时 `multiprocessing.util._exit_function` 对这些子进程调用无超时 `p.join()` → 永久挂起。

`coco` 的 plan 含 SMP stage，每个本地 worker 还会拉起 2 个 SMP actor 子进程，链上任何一环卡住都会放大成整 cell 挂死。

### 1.3 修复

- `cedar/client/utils.py`：新增 `kill_process_tree(pid)`，先用 psutil 枚举整棵子树（worker + 其 SMP actor），再逐个 SIGKILL，返回被杀 PID 列表。
- `cedar/client/dataset.py`：`MP_WORKER_EXIT_GRACE_SEC=2.0`、`MP_WORKER_TERMINATE_GRACE_SEC=5.0`、`MP_WORKER_KILL_GRACE_SEC=5.0`；`_MultiprocessDataSetIter._shutdown()` 改为分级、有界的关闭：
  1. 置 `done` + 唤醒 epoch 事件；
  2. 宽限 join（2 s）；
  3. 对仍存活的 worker 发 SIGTERM 并 join（5 s）；
  4. 仍存活的 worker **连同其进程树 SIGKILL** 并 join（5 s）；
  5. 若仍有存活者 → 抛 `RuntimeError`（让 cell 记录失败，而不是挂 2 小时）。

  每一步都用同一个截止时间，整个 shutdown 有上界；`_join_workers`/`_alive_workers` 为新增的静态辅助函数。
- 未做任何旧行为兼容；`DataSet.close()` 语义不变（调用 `_shutdown()`）。

### 1.4 验证

- `tests/test_dataset.py` 新增两例：
  - `test_mp_iter_shutdown_kills_stuck_worker_process_tree`：worker 刻意忽略 SIGTERM 并阻塞在 `queue.put()`，且带一个子进程；断言 `_shutdown()` 在 20 s 内返回、worker 与子进程都不再存活。
  - `test_mp_dataset_close_is_bounded_after_partial_consumption`：真实 `DataSet(iter_mode="mp")`，消费 10 条后 `close()`，断言 30 s 内返回（旧实现会永久挂住）。
- 两例均通过（`2 passed`）。整文件 `tests/test_dataset.py`：修复后 33 passed / 1 failed / 5 errors，未修复时（HEAD）31 passed / **同样的 1 failed / 5 errors**；这 6 个失败与本改动无关，是「本地已有 Ray 节点接入远端集群时 `ray.init(num_cpus=16)` 连不上」的环境问题（`test_optimizer_*` 在 `ray.init` 处报 `ValueError: When connecting to an existing cluster ...`）。
- 直接对照实验（旧路径 vs 新函数，`/tmp/old_shutdown_check.py`）：

```
old code: Terminating worker ... (pid=413948)
old code: worker alive after terminate+join(5) = True     <-- 旧逻辑丢下一个活着的子进程
new helper: killed pids [413949, 413948]                  <-- 新的进程树 kill
after kill_process_tree: worker alive = False
```

### 1.5 快照约定

本次**没有**改动 `outputs/ultimate_eight_optimizers_20260920/modules/`（历史快照保持不变）。恢复实验必须用修复后的代码生成**新的 root**，见 §4。

## 2. 暂停时刻的实验状态

### 2.1 cell 级状态（`wall` = runner 记录的 cell 墙钟，含启动与优化）

| 负载 | optimizer | 状态 | 墙钟 | 稳态吞吐 |
| --- | --- | --- | ---: | ---: |
| simclrv2 | cedar-opt | completed | 467 s | 1187.7 /s |
| simclrv2 | plumber-opt | completed | 1789 s | 106.9 /s |
| simclrv2 | ray-opt | completed | 2334 s | 83.7 /s |
| simclrv2 | unopti | completed | 4115 s | 46.2 /s |
| simclrv2 | old_dp_boundary | completed | 124 s | 1937.5 /s |
| simclrv2 | simple_dp_boundary | completed | 123 s | 1964.5 /s |
| simclrv2 | simple_dp_workers_width_boundary | completed | 357 s | 2501.5 /s |
| simclrv2 | simple-dp-opt | completed | 122 s | 1975.8 /s |
| simclrv2 | old-dp-opt | completed | 564 s | 352.9 /s |
| simclrv2_cache | cedar-opt | completed | 626 s | 1634.8 /s |
| simclrv2_cache | plumber-opt | completed | 1456 s | 131.5 /s |
| simclrv2_cache | ray-opt | completed | 2202 s | 88.8 /s |
| simclrv2_cache | unopti | completed | 4093 s | 46.5 /s |
| simclrv2_cache | old_dp_boundary | completed | 202 s | 2231.6 /s |
| simclrv2_cache | simple_dp_boundary | completed | 128 s | 5502.4 /s |
| simclrv2_cache | simple_dp_workers_width_boundary | completed | 524 s | 5399.4 /s |
| simclrv2_cache | simple-dp-opt | completed | 129 s | 5478.3 /s |
| simclrv2_cache | old-dp-opt | completed | 1311 s | 340.1 /s |
| commonvoice | cedar-opt | completed | 2200 s | 157.8 /s |
| commonvoice | plumber-opt | completed | 2087 s | 145.5 /s |
| commonvoice | ray-opt | completed | 3288 s | 91.9 /s |
| commonvoice | unopti | timeout | 7202 s | — |
| commonvoice | old_dp_boundary | completed | 485 s | 661.8 /s |
| commonvoice | simple_dp_boundary | completed | 451 s | 734.4 /s |
| commonvoice | simple_dp_workers_width_boundary | completed | 453 s | 743.2 /s |
| commonvoice | simple-dp-opt | completed | 464 s | 692.3 /s |
| commonvoice | old-dp-opt | completed | 3873 s | 79.9 /s |
| coco | cedar-opt | completed | 1960 s | 26.7 /s |
| coco | plumber-opt | completed | 2667 s | 19.0 /s |
| coco | ray-opt | completed | 6659 s | 7.6 /s |
| coco | unopti | timeout | 7203 s | — |
| coco | old_dp_boundary | timeout（伪超时，测量已完成） | 7202 s | 226.5 /s |
| coco | simple_dp_boundary | timeout（伪超时，测量已完成） | 7202 s | 240.0 /s |
| coco | simple_dp_workers_width_boundary | completed | 238 s | 294.3 /s |
| coco | simple-dp-opt | 未运行（暂停时正在跑，无结果） | — | — |
| coco | old-dp-opt | 未运行（暂停） | — | — |

吞吐口径与完整计划表见 `docs/experiment_results_20260920.md`（本次已刷新，新增 §1.5 coco；§1.4 commonvoice 的 W-aware 模型对照仍保留）。

### 2.2 未完成清单（恢复后需要跑的 cell）

| 负载 | 待跑 optimizer | 说明 |
| --- | --- | --- |
| coco | `simple-dp-opt` | 暂停时正在运行（21:52 UTC+8 启动，暂停前最后进度 35,998/50,000 条），无结果文件 |
| coco | `old-dp-opt` | 未开始 |
| coco | `old_dp_boundary` / `simple_dp_boundary` | 已有可用结果，但状态是伪超时，需重跑以拿到干净状态（见 §4.2） |
| llava_pretrain | 其余 7 个 | `cedar-opt`、`simple_dp_workers_width_boundary` 按用户要求跳过 |
| stackexchange | 其余 7 个 | 同上两个跳过 |

### 2.3 产物位置

- 状态与元数据：`outputs/ultimate_eight_optimizers_20260920/status.json`、`metadata.json`（暂停瞬间的快照另存 `/tmp/status_snapshot_pause.json`）
- 每个负载：`profiles/shared.yaml`（复用自 `outputs/six_workload_profile_v3_20260919`，coco profile 另含 `outputs/llava_profile_check_20260919` 的来源）、`plans/*.yaml`、`results/*.json`、`logs/*.log`、`cache/`、`warmup_results/`
- 输入子集：`inputs/llava_pretrain.jsonl`（50,000 行）、`inputs/stackexchange.jsonl`（20,000 行）
- 该 root 的 `modules/` 是 2026-09-19 20:15 的代码快照（未含本次修复）

## 3. 资源状态

- 远端 Ray 集群保持运行（head `172.23.166.105:6379`，资源 `cedar_remote`），driver 节点的本地 Ray 节点也在（session `2026-09-15_18-32-52`，`cedar_local`），可直接复用。
- 停实验时先 `SIGTERM` runner（禁止再开新 cell），再 `SIGTERM` 当前 cell 的进程组；容器内已确认**无任何存活的实验进程**（cell 进程组 0 个非僵尸进程）。
- 容器的 PID 1 不回收子进程，历史 campaign 累计留下约 9,000 个 `<defunct>` 僵尸条目（不占 CPU/内存，只占 PID 表；`pids.current≈12.3k / pids.max≈629k`，无风险）。彻底清空需要重启该容器。
- 停止方式记录：`kill -TERM 284114`（runner）、`kill -TERM -404905`（cell 进程组）。

## 4. 恢复步骤（本次未执行）

原则：**旧 root 只读**，用修复后的代码生成新 root，再把已完成的结果/计划/日志/profile 拷过去续跑。

### 4.1 生成新 root（含修复代码）

```bash
# 在容器内：docker exec optimalcedar-torch201-dev bash -lc '...'
cd /workspace/OptimalCedar && source env/bin/activate
OLD=outputs/ultimate_eight_optimizers_20260920
NEW=outputs/ultimate_eight_optimizers_fix_20260920

python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
  --output "$NEW" --prepare-only \
  --workloads simclrv2 simclrv2_cache commonvoice coco llava_pretrain stackexchange \
  --methods cedar-opt plumber-opt ray-opt unopti old_dp_boundary simple_dp_boundary \
            simple_dp_workers_width_boundary simple-dp-opt old-dp-opt \
  --repeats 1 --cell-timeout-sec 7200 \
  --layered-profile --smp-aggregate-profile \
  --simclrv2-epochs 20 --coco-split train2017 \
  --commonvoice-max-samples 300000 \
  --commonvoice-dataset-path /workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000 \
  --llava-samples 50000 --stackexchange-samples 20000 \
  --llava-source /workspace/OptimalCedar/evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl \
  --stackexchange-source /workspace/OptimalCedar/datasets/stackexchange/redpajama-stackexchange-400000.jsonl \
  --skip-cedar-workloads llava_pretrain stackexchange \
  --skip-cell simple_dp_workers_width_boundary@llava_pretrain \
  --skip-cell simple_dp_workers_width_boundary@stackexchange
```

### 4.2 迁移已完成产物 + 清理伪超时状态

```bash
for w in simclrv2 simclrv2_cache commonvoice coco; do
  mkdir -p "$NEW/$w"
  cp -a "$OLD/$w/profiles" "$OLD/$w/results" "$OLD/$w/plans" \
        "$OLD/$w/logs" "$OLD/$w/warmup_results" "$NEW/$w/"
done
cp -a "$OLD/status.json" "$OLD/metadata.json" "$NEW/"

# 关键：runner 会把 round1 的超时方法直接标成 skipped_first_round_timeout，
# 所以 coco 的两个伪超时必须从 status.json 中删掉，另外清掉被打断的 active_cell。
python - <<'PY'
import json, pathlib
new = pathlib.Path("outputs/ultimate_eight_optimizers_fix_20260920")
state = json.loads((new / "status.json").read_text())
coco = state["coco"]
drop = {"old_dp_boundary", "simple_dp_boundary"}
coco["cells"] = [c for c in coco["cells"] if c.get("method") not in drop]
coco.pop("active_cell", None)
(new / "status.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
PY
```

说明：`coco.unopti` 的 `timeout` **保留**（它是真超时：2 h 只处理 47,635/50,000，约 6.6 rec/s），恢复后会按 `skipped_first_round_timeout` 跳过。

### 4.3 续跑

> §4.1/§4.2 的机制已在临时 root `/tmp/resume_check` 上验证过（`--prepare-only` 退出码 0、快照内含 `kill_process_tree`、输入 50,000/20,000 行；状态清理后 coco 的续跑判定为 REUSE 5 个 / SKIP `unopti` / RUN `old_dp_boundary`,`simple_dp_boundary`,`simple-dp-opt`,`old-dp-opt`）。

```bash
# 在容器内 detached 启动，参数与 4.1 完全一致，只把 --prepare-only 换成 --resume
docker exec -d optimalcedar-torch201-dev bash -lc \
  'cd /workspace/OptimalCedar && source env/bin/activate && nohup python -u \
   evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
   --output outputs/ultimate_eight_optimizers_fix_20260920 --resume ... > /tmp/resume.log 2>&1 &'
```

`--resume` 会：读取 `metadata.json` 恢复参数 → 复用各负载 `profiles/shared.yaml`（状态为 `completed` 且文件存在）→ 跳过 `status.json` 里已记录的 cell → 只补跑 §2.2 的缺口。

### 4.4 恢复后要盯的点

1. coco 的 `old_dp_boundary` / `simple_dp_boundary` 重跑完应给出与旧记录一致的 ~226 / ~240 rec/s，且 cell 状态为 `completed`（不再挂 2 h）。
2. llava_pretrain / stackexchange 的 cell 里若出现 SMP stage，同样会走新的分级 shutdown；日志中应出现 `Worker N ignored SIGTERM; killing its process tree (pids [...])`，随后正常进入下一个 cell。
3. 若日志出现 `Dataset workers survived SIGKILL ...`，说明进程处于不可杀状态（通常是 D 状态 IO），需要人工介入，不要让它继续占用 CPU 预算。

## 5. 代码改动清单（本次提交）

| 文件 | 改动 |
| --- | --- |
| `cedar/client/utils.py` | 新增 `kill_process_tree(pid)`（psutil，SIGKILL 整棵子树） |
| `cedar/client/dataset.py` | `_MultiprocessDataSetIter._shutdown()` 分级有界关闭（2 s / 5 s / 5 s + 进程树 SIGKILL + 存活即报错），新增 `_join_workers` / `_alive_workers` 与三个宽限常量 |
| `tests/test_dataset.py` | 新增 2 个回归测试（忽略 SIGTERM 的卡死 worker；部分消费后 `close()` 有界） |
| `scripts/make_results_doc.py` | §1 允许包含 coco；§3 记录暂停与伪超时根因；W-aware 对照段落固化为 fragment；状态列按 runner label 正确映射 |
| `docs/experiment_results_20260920_w_models.inc.md` | 新增：commonvoice W-aware 模型对照段落（生成器引用，避免再次丢失） |
| `docs/experiment_results_20260920.md` | 重新生成：新增 §1.5 coco，修正 coco/llava 的状态列 |
