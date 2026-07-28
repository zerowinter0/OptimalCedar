from evaluation.fastflow.workloads.common import BaselineFastFlowApp


class CommonVoiceApp(BaselineFastFlowApp):
    workload = "commonvoice"


APP_CLASS = CommonVoiceApp
