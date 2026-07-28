from evaluation.fastflow.workloads.common import BaselineFastFlowApp


class WikiText103App(BaselineFastFlowApp):
    workload = "wikitext103"


APP_CLASS = WikiText103App
