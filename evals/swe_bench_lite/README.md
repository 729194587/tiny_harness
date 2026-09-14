# Single-run finalize

```bash
python -m evals.swe_bench_lite finalize <run-directory>
python -m evals.swe_bench_lite finalize <run-directory> --dataset <dataset.jsonl>
```

输入为包含 `predictions.jsonl` 和 rollout `events.jsonl` 的单任务 run
目录。数据集默认取 run metadata 的 `selected_tasks_path`，旧 run 未记录时
使用 bundled `selected_tasks.jsonl`；迁移目录后可用 `--dataset` 覆盖。
metadata 中的相对路径以 run 目录为基准。

命令复用现有 official evaluator，因此首次评测需要 SWE-bench harness 和
Docker 环境；不会调用 rollout 模型。它检查 run 内及上一级 results 目录的
`official_evaluation/*`，仅复用关联当前 predictions、退出成功且有明确 grader
结论的结果。新结果记录 predictions SHA-256；旧结果无摘要时要求 predictions
修改时间不晚于评测 metadata。修改或移动历史 artifacts 后可能需要重新评测。

输出 `summary.json`、`summary.md`，终端打印与 Markdown 相同的核心结果。
`Resolved` 和 `Unresolved` 表示有效评测结论，退出码均为 0；评测异常、非零
退出码、缺失或冲突的 grader 结论为 `Evaluation Error`，退出码 2。
输入不合法（例如多个 instance）直接报错，不启动评测。

metrics 复用 offline report：逻辑 turns 去除重试重复，usage 包含辅助模型调用，
overall cache hit rate 为累计 hit / (累计 hit + 累计 miss)，不是逐请求比例的平均。
缺失字段用 JSON null / “未知” 表示，部分 usage 的观测总量和覆盖数保留在 JSON。
checkpoint 仅统计 `llm_task_state_checkpoint` 事件，普通历史结果清理不计入。
summary cache hit rate 来自 `purpose="summary"` 的模型响应。
workspace mutation 缺少完整观测时保持未知，不从文件工具返回值推断。
