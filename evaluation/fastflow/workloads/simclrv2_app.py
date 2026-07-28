from evaluation.fastflow.workloads.common import BaselineFastFlowApp


class SimCLRv2App(BaselineFastFlowApp):
    workload = "simclrv2"


APP_CLASS = SimCLRv2App
