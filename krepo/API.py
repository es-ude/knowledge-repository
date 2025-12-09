import mlflow
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from torch.nn import Module as TorchModule
else:
    TorchModule = Any


class KnowledgeRepoAPI:
    def __init__(self, tracking_uri: str = "http://localhost"):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = None
        self.current_run = None
        self.nas_config = None
        self.search_space_config = None
        self.hw_platform = None

    def set_experiment(self, experiment_name: str, nas_config: dict, search_space_config: dict, hw_platform: str) -> None:
        """
        Specify the current experiment
        Create/GET Experiment ID
        Save following logs under this experiment
        """
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
            self.nas_config = nas_config
            self.search_space_config = search_space_config
            self.hw_platform = hw_platform
        else:
            raise RuntimeError(f"Could not find or create experiment: {experiment_name}")

    def start_run(self, run_name) -> None:
        if not self.experiment_id:
            raise RuntimeError("Experiment not set. Call set_experiment() first.")

        self.end_run()
        self.current_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

        mlflow.log_dict(self.nas_config, "nas_config.json")
        mlflow.log_dict(self.search_space_config, "search_space_config.json")

        self.set_tag("hw_platform", self.hw_platform)

    def end_run(self):
        if self.current_run:
            mlflow.end_run()
            self.current_run = None

    def set_tag(self, key: str, value: Any) -> None:
        if not self.current_run:
            raise RuntimeError("Run not started. Use start_run() first.")

        mlflow.set_tag(key, value)


    def set_tags(self, tags: Dict[str, str]) -> None:
        if not self.current_run:
            raise RuntimeError("Run not started. Use start_run() first.")

        for key, value in tags.items():
            mlflow.set_tag(key, value)


    def log_metrics(self, metrics: Dict, step=0) -> None:
        if not self.current_run:
            raise RuntimeError("Run not started. Use start_run() first.")

        mlflow.log_metrics(metrics, step=step)


    def log_model(self, parameters: Dict, model_architecture: Dict, model: TorchModule | None = None) -> None:
        if not self.current_run:
            raise RuntimeError("Run not started. Use start_run() first.")

        mlflow.log_params(parameters)
        mlflow.log_dict(model_architecture, "model_architecture.json")

        if model:
            mlflow.pytorch.log_model(model)

    # -------- Estimator Methods --------
    def get_training_data_for_estimator(self, target_metric: str, hw_platform: str, tag: List[str]) -> List[Tuple[Dict, Dict]]:
        """
        Returns all the model_architectures and target metrics that
        correspond to the given hardware platform and model type
        """
        pass

    def save_estimator(self, estimator: TorchModule, hw_platform: str, tags: List[str], metric: str) -> None:
        pass

    def load_estimator(self, hw_platform: str, tags: List[str], metric: str) -> TorchModule:
        pass
