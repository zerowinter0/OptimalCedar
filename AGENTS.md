# Project Instructions

运行任何代码前使用 `source env/bin/activate` 切换环境。更多运行要求可见 `README.md`

本项目基于CMU的Cedar(VLDB 2024)，目的为在此基础上进行改进（将原Cedar的optimizer.py改为dp_optimizer.py（或my_optimizer.py，两者逻辑一样，不一致处以my_optimizer.py为准）。优化处：1.将原先耗时的重排部分使用DP算法重新实现，耗时更小。2.原先的多种优化为分阶段优化，现在使用统一的DP进行联合优化，因此优化空间更大，最终优化结果的cost更低。

我可能会提出对代码的改进要求，论文的文字修改要求，实验运行要求。

代码路径为/OptimalCedar/cedar，论文路径为/OptimalCedar/paper/OptimalReorder。此外，/OptimalCedar/paper/other_paper_examples提供了一些与本项目研究内容相近的一些论文，可以参考它们进行写作