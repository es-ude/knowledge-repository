import mlflow
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import json
import os

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
        experiment = mlflow.get_experiment_by_name(experiment_name)
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


    def set_tags(self, tags: Dict[str, Any]) -> None:
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
    def get_training_data_for_estimator(self, target_metric: str, hw_platform: str, tags: Dict[str, Any] | None = None) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Returns all the model_architectures and target metrics that
        correspond to the given hardware platform and model type
        """
        filter_parts = [f'tags."hw_platform" = "{hw_platform}"']

        if tags:
            for key, value in tags.items():
                filter_parts.append(f'tags."{key}" = "{value}"')

        filter_string = ' AND '.join(filter_parts)

        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=filter_string,
        )

        training_data: List[Tuple[Dict, Dict]] = []
        client = mlflow.tracking.MlflowClient()

        for _, run in runs.iterrows():
            run_id = run["run_id"]
            metric_value = run[f"metrics.{target_metric}"]
            if metric_value is None:
                continue

            try:
                artifact_path = "model_architecture.json"
                file_path = client.download_artifacts(run_id=run_id, path=artifact_path)
                print("Test")
                # FIXME: Downloading, Reading and Loading artifacts takes time.
                #        Considering storing model_architecture as tag instead? (Considering SQL)
                with open(file_path, 'r') as f:
                    model_architecture = json.load(f)

                target_metric_dict = {target_metric: metric_value}
                training_data.append((model_architecture, target_metric_dict))

                if os.path.exists(file_path):
                    os.remove(file_path)

            except Exception as e:
                print(f"Skipping run '{run_id}' due to error: {e}")
                continue

        return training_data


    def save_estimator(self, estimator: TorchModule, hw_platform: str, metric: str, validation_loss: float, tags: Dict[str, Any] | None = None) -> str:
        """
        Save this estimator to a new run and return a UID for easy access using krepo.API.get_model_by_uid()
        """
        if self.current_run:
            self.end_run()

        self.start_run("Estimator")

        self.set_tag("model_type", "estimator")
        self.set_tag("metric", metric)
        self.set_tag("hw_platform", hw_platform)

        self.log_metrics(
            {"validation_loss": validation_loss}
        )

        if tags:
            self.set_tags(tags)

        model_info: mlflow.models.model.ModelInfo = mlflow.pytorch.log_model(
            pytorch_model=estimator,
            artifact_path="estimator_model"
        )
        print(f"Estimator saved to MLflow run {self.current_run.info.run_id} under 'estimator_model'")
        model_uid = model_info.model_id

        return model_uid


    def load_estimator(self, hw_platform: str, metric: str, tags: Dict[str, Any] | None = None) -> TorchModule:
        if not self.experiment_id:
            raise RuntimeError("Experiment not set. Call set_experiment() first.")

        filter_parts = [
            f'tags."hw_platform" = "{hw_platform}"',
            f'tags."model_type" = "estimator"',
            f'tags."metric" = "{metric}"'
        ]

        if tags:
            for key, value in tags.items():
                filter_parts.append(f'tags."{key}" = "{value}"')

        filter_string = " AND ".join(filter_parts)

        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=["metrics.validation_loss ASC"],
            max_results=1
        )

        if not runs.empty:
            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/estimator_model"
            estimator = mlflow.pytorch.load_model(model_uri)
            return estimator
        else:
            raise LookupError(f"No estimator found matching criteria: HW='{hw_platform}', Metric='{metric}', Tags='{tags}'")


def get_model_by_uid(uid: str) -> TorchModule:
    model: TorchModule = mlflow.get_logged_model(uid)
    return model
