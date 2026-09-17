"""Summarize all repeats, preserving failed and censored cells."""
import csv
import json
import statistics
import sys
from pathlib import Path


def summarize(root):
    status_path = root/'status.json'
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    rows = []
    for workload, progress in status.items():
        for label, internal in metadata['methods'].items():
            records = [r for r in progress['cells'] if r['method']==label]
            walls, throughputs, setups, workers = [], [], [], []
            for record in records:
                if record['status'] != 'completed':
                    continue
                file = root/workload/'results'/f"round{record['round']}__{internal}.json"
                if not file.exists():
                    continue
                payload = json.loads(file.read_text())
                for run in payload.get('runs', []):
                    wall = run.get('workload_wall_time_sec',0)
                    if wall <= 0:
                        continue
                    walls.append(wall)
                    setups.append(run['setup_time_sec'])
                    # Input records/sec compares filter pipelines fairly even
                    # when optimizer order changes the surviving output count.
                    count = metadata['input_records'][workload]
                    throughputs.append(count/wall)
                    plans = run.get('physical_plans_by_feature',{}).values()
                    workers.append(next(iter(plans),{}).get('n_local_workers'))
            rows.append(dict(workload=workload, method=label, completed=len(walls),
                statuses=';'.join(r['status'] for r in records),
                W=';'.join(str(w) for w in workers),
                mean_input_records_per_sec=statistics.mean(throughputs) if throughputs else None,
                stdev_input_records_per_sec=statistics.stdev(throughputs) if len(throughputs)>1 else None,
                mean_execution_sec=statistics.mean(walls) if walls else None,
                mean_setup_sec=statistics.mean(setups) if setups else None))
    with (root/'summary.csv').open('w') as out:
        writer=csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ['workload'])
        writer.writeheader();writer.writerows(rows)
    text=['# Simple-DP 独立消融实验', '',
          f'吞吐量按完整输入条数 / 正式遍历 wall time 计算，包含启动和排空，排除优化及 cache 预热。未完成配置的 {metadata["repeats"]} 轮时不作为完整均值。', '',
          '| 负载 | 方法 | 成功轮数 | 输入条/秒（均值 ± 标准差） | 优化/构建秒 | 状态 |',
          '|---|---|---:|---:|---:|---|']
    for row in rows:
        mean=row['mean_input_records_per_sec'];sd=row['stdev_input_records_per_sec']
        rate='—' if mean is None else f'{mean:.2f} ± {sd:.2f}' if sd is not None else f'{mean:.2f}'
        setup=row['mean_setup_sec'];setup='—' if setup is None else f'{setup:.2f}'
        text.append(f"| {row['workload']} | {row['method']} | {row['completed']}/{metadata['repeats']} | {rate} | {setup} | {row['statuses']} |")
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    summarize(Path(sys.argv[1]))
