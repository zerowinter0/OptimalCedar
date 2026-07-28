# Project Instructions

你当前处在实验室服务器中。
运行或修改任何代码前需进入代码实际处在的Docker容器中，然后使用 `source env/bin/activate` 切换环境。更多运行要求可见 `README.md`。

本项目基于CMU的Cedar(VLDB 2024)，目的为在此基础上进行改进（将原Cedar的optimizer.py改为dp_optimizer.py（或my_optimizer.py，两者逻辑一样，不一致处以my_optimizer.py为准）。优化处：1.将原先耗时的重排部分使用DP算法重新实现，耗时更小。2.原先的多种优化为分阶段优化，现在使用统一的DP进行联合优化，因此优化空间更大，最终优化结果的cost更低。

我可能会提出对代码的改进要求，论文的文字修改要求，实验运行要求。

对于代码或实验的任何修改都需要以一个正式顶会工作为目标，例如不可为了让实验正常运行而使用调低部分测试的参数等方式使得实验失去公平性。

对于运行长时间实验的要求，请使用nohup等方式进行离线实验并使用日志文件记录输出，无需实时监控实验情况。

代码路径为/OptimalCedar/cedar。

论文路径为/my_paper/

其他参考论文路径为/other_paper_examples，其中部分论文的文字已经提取到extracted_text中，无需再次读取pdf。

## 当前实验

- 负载：`coco`、`commonvoice`、`commonvoice_cache`、`llava_pretrain`、`redpajama_c4`、`simclrv2`、`simclrv2_cache`、`wikitext103`、`wikitext103_cache`；非 cache 负载关闭 cache，`*_cache` 开启全部优化。
- 每个负载均比较 `optimizer`、`dj_optimizer`、`dp_cedar_optimizer`、`dp_optimizer`、`dp_two_stage_optimizer`，使用同一份 profile；profile 阶段每个 Ray/SMP stage 为 1 actor/process、运行 10 秒。
- 固定 `W=8`、`CPU_BUDGET=64`，所有 optimizer 使用相同数据量、开关和重复次数。
- 控制数据量：LLaVA 为 20,000 samples；RedPajama-C4 约 2 GiB（829,916 samples）；其余负载输入不超过 3 GiB（SimCLRv2 9,469 files，WikiText-103 1,801,350 lines）。
- cache 负载先独立预热生成 cache，再正式运行；仅记录正式运行。每个 optimizer 运行 3 次并 round-robin 轮换顺序。
- 每次启动清理所选负载的 `plans/results/warmup_results/logs/cache`，重新物化计划；单个 optimizer 的计划优化超过 5 分钟即记为不可用并跳过。
- 长实验用 `nohup` 离线运行；每个负载目录保留 profile、计划、日志、metadata 和 JSON 结果。可通过 `run_w8_plan_and_matrix.sh --workloads ...` 仅运行指定负载。
