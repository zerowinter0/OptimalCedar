"""Generate the campaign result tables (docs/experiments.md §1-§2) from JSON/YAML.

The consolidated experiment document is hand-maintained; this script only
regenerates the campaign tables so they can be diffed against the frozen copy
inside `docs/experiments.md`.  Default output: `outputs/experiment_tables.md`,
override with `--out PATH`.
"""
import collections
import json
import pathlib
import sys
import time
import yaml

REPO = pathlib.Path("/workspace/OptimalCedar")
ULT = REPO / "outputs/ultimate_eight_optimizers_fix_20260921"
SMALL = REPO / "outputs/six_workload_formal_v3_20260919"
W_MODEL_FRAGMENT = REPO / "scripts/templates/commonvoice_w_models.inc.md"

ULT_METHODS = [
    ("optimizer", "cedar-opt"), ("plumber_optimizer", "plumber-opt"),
    ("raydata_optimizer", "ray-opt"), ("unopti", "unopti"),
    ("old_dp_boundary", "dp-boundary"),
    ("simple_dp_boundary", "dp-boundary-affine"),
    ("simple_dp_workers_width_boundary", "dp-boundary-affine-W-width"),
    ("simple_dp", "simple-dp-opt (new profile, no boundary)"),
    ("old_dp_legacy_optimizer", "old-dp-opt (legacy profile, no boundary)"),
]
SMALL_METHODS = [
    ("optimizer", "cedar-opt"), ("plumber_optimizer", "plumber-opt"),
    ("raydata_optimizer", "ray-opt"), ("old_dp_boundary", "dp-boundary"),
    ("simple_dp_boundary", "dp-boundary-affine"),
    ("simple_dp_workers_width_boundary", "dp-boundary-affine-W-width"),
]
ULT_DATA = {"simclrv2": "189,380 (9,469 张 × 20 epoch)", "simclrv2_cache": "189,380 (= simclrv2)",
            "commonvoice": "300,000", "coco": "50,000 (train2017)",
            "llava_pretrain": "43,940 (输入 50,000，过滤后)", "stackexchange": "7,238 (输入 20,000，过滤后)"}
SMALL_DATA = {"simclrv2": "9,469", "simclrv2_cache": "9,469", "commonvoice": "15,000",
              "coco": "5,000 (val2017)", "llava_pretrain": "1,000 / 907 processed", "stackexchange": "2,000"}
DONE = [
    "simclrv2",
    "simclrv2_cache",
    "commonvoice",
    "coco",
    "llava_pretrain",
    "stackexchange",
]
# status.json records cells by the runner's method label, while results and
# plans are named after the internal optimizer module.
STATUS_LABELS = {
    "optimizer": "cedar-opt",
    "plumber_optimizer": "plumber-opt",
    "raydata_optimizer": "ray-opt",
    "unopti": "unopti",
    "old_dp_boundary": "old_dp_boundary",
    "simple_dp_boundary": "simple_dp_boundary",
    "simple_dp_workers_width_boundary": "simple_dp_workers_width_boundary",
    "simple_dp": "simple-dp-opt",
    "old_dp_legacy_optimizer": "old-dp-opt",
}


def metrics(root, workload, method):
    path = root / workload / "results" / f"round1__{method}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    run = (data.get("runs") or [{}])[0]
    return {
        "samples": run.get("num_samples"),
        "steady": run.get("perf_time_sec"),
        "throughput": run.get("throughput_samples_per_sec"),
        "setup": run.get("setup_time_sec"),
        "total": run.get("total_time_sec"),
    }


def status_of(root, workload, method):
    try:
        state = json.loads((root / "status.json").read_text())
    except Exception:
        return None
    for cell in state.get(workload, {}).get("cells", []):
        label = cell.get("method")
        if label in (method, method.replace("_optimizer", "")) or label == method:
            return cell.get("status")
    for key, alias in ULT_METHODS + SMALL_METHODS:
        if key == method and cell_for(state, workload, alias):
            return cell_for(state, workload, alias)
    return None


def cell_for(state, workload, label):
    for cell in state.get(workload, {}).get("cells", []):
        if cell.get("method") == label:
            return cell.get("status")
    return None


def plan_lines(path):
    """Render one plan as (W, chain, cache placement)."""
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    features = sorted(data)
    plan = data[features[0]]
    pipes = plan["pipes"]
    graph = {int(k): (int(v) if str(v) != "" else None) for k, v in plan["graph"].items()}
    preds = collections.Counter(v for v in graph.values() if v is not None)
    starts = [node for node in graph if preds[node] == 0]
    order, node = [], starts[0]
    while node is not None:
        order.append(node)
        node = graph.get(node)
    stages, cache_after = [], None
    for pid in order:
        rec = pipes.get(str(pid), {})
        name = (rec.get("name") or f"pipe{pid}").replace("MapperPipe_", "").replace("FilterPipe_", "")
        ctx = rec.get("variant_ctx") or {}
        variant = rec.get("variant") or ctx.get("variant_type") or "?"
        fused = rec.get("fused_pipes") or []
        width = ctx.get("n_actors") or ctx.get("n_procs") or ""
        tag = name if not fused else f"{name}{{{','.join(str(x) for x in fused)}}}"
        if variant not in ("INPROCESS", None):
            tag = f"{tag}[{variant} w={width}]"
        if "Cache" in name:
            cache_after = len(stages)
        stages.append(tag)
    cache = "无"
    if cache_after is not None:
        where = stages[cache_after - 1] if cache_after else "开头"
        cache = f"有（在 {where} 之后）"
    return plan.get("n_local_workers"), " -> ".join(stages), cache, len(features)


def metric_table(root, workload, methods, data_note):
    lines = ["| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    rows = []
    baseline = None
    state_json = json.loads((root / "status.json").read_text())
    for key, label in methods:
        m = metrics(root, workload, key)
        state = cell_for(state_json, workload, STATUS_LABELS.get(key, key))
        if m is None:
            rows.append((label, None, state))
            continue
        if baseline is None and key == "optimizer":
            baseline = m["throughput"]
        rows.append((label, m, state))
    for label, m, state in rows:
        if m is None:
            lines.append(f"| {label} | {data_note} | — | — | — | — | — | {state or '未运行'} |")
            continue
        ratio = (m['throughput'] / baseline) if (baseline and m['throughput']) else None
        lines.append("| %s | %s | %.1f s | %.1f /s | %s | %.1f s | %.1f s | %s |" % (
            label, f"{m['samples']:,}" if m['samples'] else data_note,
            m["steady"] or 0.0, m["throughput"] or 0.0,
            f"{ratio:.2f}×" if ratio else "—", m["setup"] or 0.0, m["total"] or 0.0,
            state or "completed"))
    return lines


def plan_table(root, workload, methods):
    lines = ["| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |", "| --- | ---: | --- | --- |"]
    for key, label in methods:
        rendered = plan_lines(root / workload / "plans" / f"round1__{key}.yaml")
        if rendered is None:
            lines.append(f"| {label} | — | 无计划文件（未运行/超时） | — |")
            continue
        w, chain, cache, nfeat = rendered
        lines.append(f"| {label} | {w} | {chain} | {cache} |")
    return lines


out = [
    "# 实验结果汇总（2026-09-20）",
    "",
    "本文件由脚本从 `outputs/` 中的实验记录自动生成，包含：",
    "1) 放大数据集 campaign（9 个 optimizer）；",
    "2) 之前小数据集 campaign（6 个 optimizer）的对照结果；",
    "3) 每个 optimizer 实际选中的物理计划。",
    "",
    "## 0. 实验设置",
    "",
    "### 0.1 optimizer",
    "",
    "| 标签 | 实现 (selector) | 说明 |",
    "| --- | --- | --- |",
    "| cedar-opt | Optimizer (0) | Cedar 原始分阶段优化器 |",
    "| plumber-opt | PlumberOptimizer (18) | Plumber 基线 |",
    "| ray-opt | RayDataOptimizer (19) | Ray Data 基线 |",
    "| unopti | UnoptimizedOptimizer (24) | 未优化，直接执行原计划 |",
    "| dp-boundary | OldDpBoundaryOptimizer (28) | 只用 boundary，计价沿用 Cedar 旧 profile 属性 |",
    "| dp-boundary-affine | SimpleDpBoundaryOptimizer (21) | boundary + 每个算子 kx+b（新 layered profile） |",
    "| dp-boundary-affine-W-width | SimpleDpWorkersWidthBoundaryOptimizer (27) | boundary + kx+b + Workers + 宽度搜索 |",
    "| simple-dp-opt | LayeredSimpleDpOptimizer (29) | 与 dp-boundary-affine 相同，但**不加 stage boundary 项** |",
    "| old-dp-opt | SimpleDpOptimizer (11) | 与 simple-dp-opt 相同，但只读 Cedar 旧 profile 属性 |",
    "",
    "### 0.2 数据量与协议",
    "",
    "| 负载 | 放大 campaign | 小数据集 campaign |",
    "| --- | ---: | ---: |",
    "| simclrv2 | 189,380（本地 9,469 张重复 20 遍） | 9,469 |",
    "| simclrv2_cache | 189,380（同上，cache 全开） | 9,469 |",
    "| commonvoice | 300,000 | 15,000 |",
    "| coco | 50,000（train2017） | 5,000（val2017） |",
    "| llava_pretrain | 50,000 | 1,000（实际处理 907） |",
    "| stackexchange | 20,000 | 2,000 |",
    "",
    "- 远端 Ray（`cedar_remote`）执行，`CPU_BUDGET=64`，非 cache 负载关闭 cache、`*_cache` 全开；",
    "- 放大 campaign：单 cell 上限 2 小时，超时记 unavailable 并继续；`cedar-opt` 与 `dp-boundary-affine-W-width` 在 llava/stackexchange 上跳过；",
    "- 小数据集 campaign：单 cell 上限 1 小时，`cedar-opt` 在 llava/stackexchange 上跳过；",
    "- 指标：稳态吞吐 = 总数据量 / 稳态时间；非稳态 setup 含 Ray 启动与计划优化；总时长 = setup + 数据执行墙钟。",
    "",
    "## 1. 放大数据集 campaign（`outputs/ultimate_eight_optimizers_20260920`）",
    "",
]
section = 0
for workload in DONE:
    section += 1
    out.append(f"### 1.{section} {workload}")
    out.append("")
    out.append(f"- 数据量：{ULT_DATA[workload]}")
    out.append("")
    out.append("**结果**")
    out.append("")
    out += metric_table(ULT, workload, ULT_METHODS, ULT_DATA[workload])
    out.append("")
    out.append("**各 optimizer 选中的计划**")
    out.append("")
    out += plan_table(ULT, workload, ULT_METHODS)
    out.append("")
    if workload == "commonvoice":
        # Plumber/PICO W-aware model comparison; the fragment is versioned so
        # regenerating this document cannot drop it.
        section += 1
        out.append(W_MODEL_FRAGMENT.read_text().format(no=section).rstrip())
        out.append("")

out += [
    "## 2. 小数据集 campaign（`outputs/six_workload_formal_v3_20260919`）",
    "",
    "该轮为放大前的对照实验（同样 1 轮、CPU_BUDGET=64、远端 Ray）。",
    "",
]
for workload in ["simclrv2", "simclrv2_cache", "commonvoice", "coco", "llava_pretrain"]:
    out.append(f"### 2.{['simclrv2','simclrv2_cache','commonvoice','coco','llava_pretrain'].index(workload) + 1} {workload}")
    out.append("")
    out.append(f"- 数据量：{SMALL_DATA[workload]}")
    out.append("")
    out += metric_table(SMALL, workload, SMALL_METHODS, SMALL_DATA[workload])
    out.append("")
    out.append("**各 optimizer 选中的计划**")
    out.append("")
    out += plan_table(SMALL, workload, SMALL_METHODS)
    out.append("")

state = json.loads((ULT / "status.json").read_text())
out += [
    "## 3. 未完成与不可用记录",
    "",
    "- 放大 campaign 已完成（修复 teardown 后于 `outputs/ultimate_eight_optimizers_fix_20260921` 续跑，2026-09-21 08:07（UTC+8）写出 COMPLETE）：54 个 cell 中 48 个 completed、2 个真超时、4 个按规则跳过；",
    "- `commonvoice` 的 `unopti` 超过 2 小时上限，记为 `timeout`（unavailable）；",
    "- `coco` 的 `unopti` 真的慢：2 小时内只处理 47,635/50,000（约 6.6 rec/s），记为 `timeout`；",
    "- `coco` 的 `dp-boundary` / `dp-boundary-affine` 在修复前曾在 teardown 阶段挂住 2 小时被 runner 杀掉（根因：本地 worker 阻塞在 `result_queue.put()` 后忽略 SIGTERM，解释器退出时无超时 join 子进程）；`cedar/client/dataset.py` 的分级 shutdown（进程树 SIGKILL + 有界 join）修复后重跑，分别 270 s / 264 s 正常收尾，吞吐 230.8 / 241.1 rec/s；",
    "- `llava_pretrain` 与 `stackexchange` 的 `cedar-opt` 按用户要求跳过，`dp-boundary-affine-W-width`（PICO）因已知的超时记为 `skipped_previous_timeout`（见 §4 的复杂度分析）；",
    "- 输入记录数 vs 实际处理量：`llava_pretrain` 配置 50,000 实际处理 43,940，`stackexchange` 配置 20,000 实际处理 7,238 —— 两个 pipeline 内含 `FilterPipe`（文本质量/语言过滤、图像存在性等），被过滤的记录不进入统计；同一负载内所有 optimizer 处理量一致，横向比较仍然公平；",
    "- 小数据集 campaign 的 `llava_pretrain`：`cedar-opt` 按用户要求跳过，`dp-boundary-affine-W-width` 因 1 小时上限记为 `timeout`（单次 W 的精确 DP 在 16 层中的第 10 层被截断）；",
    "- 小数据集 campaign 的 `stackexchange` 按用户要求提前停止（只完成 plumber-opt / ray-opt），本文件不将其计入对照。",
    "",
    "## 4. 观察",
    "",
    "- **cache 负载**：只有 DP 系列会插入 `ObjectDiskCachePipe`（均落在 ImageReader 之后、增广算子之前），cedar/plumber/raydata 虽然 cache 已开启但没有落盘策略；simclrv2_cache 上 `simple-dp-opt` / `dp-boundary-affine` 达到 ~5.5k rec/s，是 cedar-opt 的 3.4–3.6 倍。",
    "- **放大后排序稳定**：simclrv2 上 dp-boundary-affine-W-width (2,502/s) > simple-dp-opt / dp-boundary-affine (~1,970/s) > dp-boundary (1,938/s) > cedar-opt (1,188/s) > old-dp-opt (353/s)，而未优化计划只有 46/s。",
    "- `old-dp-opt`（旧 profile 属性、无 boundary）明显差于新 profile 的 `simple-dp-opt`（353 vs 1,976 /s），说明新 profile 的隔离测量 + kx+b 是主要收益来源。",
    "- 计划形态：DP 系列倾向把 CPU 段整体融合并选大 W；cedar-opt 保留一个 RAY stage；plumber/raydata 把算子拆成多个 SMP/RAY stage（radata 在 coco 上把全部算子塞进单个 RAY w=64，吞吐仅 4 rec/s）。",
    "",
]
doc = "\n".join(out) + "\n"
target = pathlib.Path(
    sys.argv[sys.argv.index("--out") + 1]
    if "--out" in sys.argv
    else REPO / "outputs/experiment_tables.md"
)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(doc)
print(f"wrote {target} ({len(doc.splitlines())} lines)")
