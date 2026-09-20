import os,time,json,pickle,statistics,multiprocessing as mp
from pathlib import Path
import numpy as np
from cedar.service.actor import SMPActor
from cedar.pipes.common import DataSample
root=Path(os.environ.get("CEDAR_SMP_TRANSPORT_OUTPUT", "/tmp/cedar_smp_transport"))
root.mkdir(parents=True, exist_ok=True)

class IdentityActor(SMPActor):
    def process(self,data):
        if isinstance(data,tuple) and data[0]=="produce":
            return np.full(data[1]//4,1,dtype=np.float32)
        return data

def pair_driver(index,size,mode,window,ready,start,result):
    requests=mp.Queue();responses=mp.Queue()
    actor=IdentityActor("transport_probe",disable_torch_parallelism=False)
    actor.register(requests,responses);actor.start()
    payload=("produce",size) if mode=="output" else np.zeros(size//4,dtype=np.float32)
    sample=DataSample(payload)
    output=DataSample(np.zeros(size//4,dtype=np.float32))
    bytes_per_record=len(pickle.dumps(sample))+len(pickle.dumps(output))
    def exchange():
        for j in range(window):requests.put(sample)
        for j in range(window):
            value=responses.get(timeout=30)
            assert value.data.nbytes==size
    try:
        for j in range(3):exchange()
        ready.put(index);start.wait(timeout=120)
        runs=[]
        for repeat in range(3):
            tick=time.perf_counter();deadline=tick+1.0;count=0
            while time.perf_counter()<deadline:
                exchange();count+=window
            finished=time.perf_counter()
            runs.append({"samples":count,"seconds":finished-tick,
                         "start":tick,"end":finished})
            # Let slow peers finish before the next repeat via the main driver.
            result.put({"index":index,"repeat":repeat,"runs":runs[-1],
                        "serialized_bytes_per_record":bytes_per_record})
            start.clear()
            # Independent per-pair events are set by the parent for each repeat.
            if repeat<2:start.wait(timeout=120)
    finally:
        actor.stop();actor.join(timeout=5)
        if actor.is_alive():actor.terminate();actor.join()
        requests.close();requests.cancel_join_thread()
        responses.close();responses.cancel_join_thread()

def measure(pairs,size,mode,window):
    ready=mp.Queue();result=mp.Queue();events=[mp.Event() for _ in range(pairs)]
    drivers=[mp.Process(target=pair_driver,args=(i,size,mode,window,ready,events[i],result))
             for i in range(pairs)]
    try:
        for p in drivers:p.start()
        for _ in drivers:ready.get(timeout=120)
        runs=[]
        for repeat in range(3):
            for event in events:event.set()
            rows=[result.get(timeout=120) for _ in drivers]
            assert all(row["repeat"]==repeat for row in rows)
            elapsed=max(row["runs"]["end"] for row in rows)-min(row["runs"]["start"] for row in rows)
            samples=sum(row["runs"]["samples"] for row in rows)
            wire_bytes=sum(row["runs"]["samples"]*row["serialized_bytes_per_record"] for row in rows)
            runs.append({"samples":samples,"elapsed_sec":elapsed,
                         "samples_per_sec":samples/elapsed,
                         "serialized_bytes_per_sec":wire_bytes/elapsed})
            # Every driver clears its event after recording a repeat. Wait for
            # that acknowledgement before releasing the next repeat.
            while any(event.is_set() for event in events):time.sleep(0.005)
        return {"pairs":pairs,"payload_bytes":size,"mode":mode,"window":window,
                "cpu_execution_processes":2*pairs,"runs":runs,
                "median_samples_per_sec":statistics.median(x["samples_per_sec"] for x in runs),
                "median_serialized_bytes_per_sec":statistics.median(x["serialized_bytes_per_sec"] for x in runs)}
    finally:
        for p in drivers:
            p.join(timeout=10)
            if p.is_alive():p.terminate();p.join()
        ready.close();ready.cancel_join_thread()
        result.close();result.cancel_join_thread()

if __name__=="__main__":
    mp.set_start_method("fork")
    report={"method":"independent Cedar SMPActor/DataSample/Queue pairs",
            "timing_excludes_startup":True,"measured_repeats":3,
            "duration_per_repeat_sec":1,"cpu_affinity":sorted(os.sched_getaffinity(0)),"tests":[]}
    for size,mode,pair_counts in [(395080,"output",[1,2,4,8,16,32]),(512,"roundtrip",[1,8,32])]:
        for window in [1,10]:
            for pairs in pair_counts:
                row=measure(pairs,size,mode,window)
                report["tests"].append(row)
                (root/"smp_report.json").write_text(json.dumps(report,indent=2))
                print("MEASURED",pairs,size,mode,window,
                      round(row["median_serialized_bytes_per_sec"]/1e6,2),
                      round(row["median_samples_per_sec"],2),flush=True)
    (root/"COMPLETE").write_text("SMP transport probe completed.\n")
