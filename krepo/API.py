import mlflow
from typing import List, Dict


class KnowledgeRepoAPI:
    def __init__(self, tracking_uri: str = "./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = None
        self.current_run = None

    def set_experiment(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            raise RuntimeError(f"Could not find or create experiment: {experiment_name}")

    def start_run(self,
                  nas_config: dict,
                  search_space_config: dict,
                  tags: List[Dict[str, str]],
                  run_name: str = None):

        if not self.experiment_id:
            raise RuntimeError("Experiment not set. Call set_experiment() first.")

        self.current_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

        log_run_configs(nas_config, search_space_config)
        set_run_tags(tags)

        return self.current_run

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        if self.current_run:
            mlflow.log_metrics(metrics, step=step)

    def log_model_architecture(self, model_type: str, model_architecture: Dict[str, dict]):
        if not self.current_run:
            raise RuntimeError("Run not started. Use start_run() first.")

        mlflow.log_param("model_type", model_type)
        mlflow.log_dict(model_architecture, "model_architecture.json")

    def end_run(self):
        if self.current_run:
            mlflow.end_run()
            self.current_run = None


def log_run_configs(nas_config: dict, search_space_config: dict):
    mlflow.log_dict(nas_config, "nas_config.json")
    mlflow.log_dict(search_space_config, "search_space_config.json")


def set_run_tags(tags: List[Dict[str, str]]):
    for tag in tags:
        mlflow.set_tag(tag["key"], tag["value"])
