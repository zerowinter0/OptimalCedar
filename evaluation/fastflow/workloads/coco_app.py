from evaluation.fastflow.workloads.common import BaselineFastFlowApp


class CocoApp(BaselineFastFlowApp):
    workload = "coco"


APP_CLASS = CocoApp
