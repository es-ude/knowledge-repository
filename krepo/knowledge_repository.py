
import mlflow
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn


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

        _log_run_configs(nas_config, search_space_config)
        _set_run_tags(tags)

        return self.current_run

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        if self.current_run:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model_type: str, model_architecture: Dict[str, dict], model: torch.nn.Module):
    # Save torch.nn module object
        if not self.current_run:
            raise RuntimeError("Run not started. Use start_run() first.")

        mlflow.log_param("model_type", model_type)
        mlflow.log_dict(model_architecture, "model_architecture.json")

    def end_run(self):
        if self.current_run:
            mlflow.end_run()
            self.current_run = None


def _log_run_configs(nas_config: dict, search_space_config: dict):
    mlflow.log_dict(nas_config, "nas_config.json")
    mlflow.log_dict(search_space_config, "search_space_config.json")


def _set_run_tags(tags: Dict[str, str]):
    for key, value in tags.items():
        mlflow.set_tag(key, value)

""""
KnowledgeRepoAPI
    -> Mlflow covers most of the API requirements except deduplication

    set_experiment(experiment_name: str, nas_config: dict, search_space_config: dict, hw_platform: str)
        -> specify the current experiment
        -> Create Experiment ID (Internally)
        -> save the following logs under this experiment
    start_run()
        -> log_metrics(metrics: dict)       (log_intermediate(metrics: dict) )
            - e.g. metrics = {"metric1" : value1, "metric2": value2, ...}
        -> log_model(model_type: str, model_architecture: dict[dict])
            - model_architecture = {model: {n_layer: int, layer_width: list[int], activation_function: str, quantisation: int}}
    get_training_data_for_estimator(target_metric: str, hw_platform: str, model_type: str) -> list[tuple[model_architecture, target_metric]]
        -> should return all the model_architectures and target metrics that corresponds to the given hardware platform and model type
    save_estimator(estimator: nn.Module, hw_platform: str, model_type: str, metric: str)
    load_estimator(hw_platform: str, model_type: str, metric: str) -> nn.Module

"""