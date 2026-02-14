import mlflow
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
import subprocess
import platform
import shutil
import json
import os

if TYPE_CHECKING:
    from torch.nn import Module as TorchModule
else:
    TorchModule = Any


class KnowledgeRepoAPI:
    def __init__(self, server_ip: str, server_port: int):
        self.tracking_uri = f"http://localhost:{server_port}"
        self.server_ip = server_ip
        self.server_port = server_port

        self.experiment_id = None
        self.current_run = None
        self.nas_config = None
        self.search_space_config = None
        self.hw_platform = None

        self._forward_port()
        mlflow.set_tracking_uri(self.tracking_uri)


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

    @staticmethod
    def load_model_from_uid(uid: str) -> TorchModule:
        model: TorchModule = mlflow.get_logged_model(uid)
        return model

    # -------- Estimator Methods --------
    @staticmethod
    def get_training_data_for_estimator(target_metric: str, hw_platform: str, tags: Dict[str, Any] | None = None) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Returns all the model_architectures and target metrics that
        correspond to the given hardware platform and model type
        """
        filter_parts = [f'tags."hw_platform" = "{hw_platform}"', 'tags."model_type" != "estimator"']

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
                print(f"Run ID: {run_id}, Artifact Path: {artifact_path}")
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


    def save_estimator(self, estimator: TorchModule, hw_platform: str, metric: str, validation_loss: float, tags: Dict[str, Any] | None = None) -> str | None:
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

        model_info = self.save_estimator_with_mlflow(estimator)
        model_uid = model_info.model_id

        print(f"Estimator saved to MLflow run {self.current_run.info.run_id} under 'estimator_model'")
        return model_uid

    @staticmethod
    def save_estimator_with_mlflow(estimator) -> mlflow.models.model.ModelInfo:
        model_info = mlflow.pytorch.log_model(
            pytorch_model=estimator,
            artifact_path="estimator_model"
        )
        return model_info

    def load_estimator(self, hw_platform: str, metric: str, tags: Dict[str, Any] | None = None) -> TorchModule:
        """
            Retrieves and loads the best-performing PyTorch estimator model
            based on hardware platform, metric type and optional tags.

            Args:
                hw_platform: The target hardware architecture
                metric: Performance metric that the estimator targets
                tags: Optional dictionary of additional tags

            Returns:
                TorchModule: The loaded PyTorch model

            It may look like there are better ways of loading the estimator,
            but I tried them and this is the only one that works.
        """
        if not self.experiment_id:
            raise RuntimeError("Experiment not set. Call set_experiment() first.")

        filter_parts = [
            f"tags.hw_platform = '{hw_platform}'",
            "tags.model_type = 'estimator'",
            f"tags.metric = '{metric}'"
        ]

        if tags:
            for key, value in tags.items():
                filter_parts.append(f"tags.{key} = '{value}'")

        filter_string = " AND ".join(filter_parts)

        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=filter_string,
            order_by=["metrics.validation_loss ASC"],
            max_results=1
        )

        all_models = self.get_all_models()

        if all_models and not runs.empty:
            run_id = runs.iloc[0]["run_id"]

            for model in all_models:
                if model.source_run_id == run_id:
                    model_uri = model.artifact_location
                    break
            else:
                raise LookupError(f"No model found matching criteria: {filter_parts[0]}")

            estimator = mlflow.pytorch.load_model(model_uri)

            return estimator

        else:
            raise LookupError(f"No estimator found matching criteria: HW='{hw_platform}', Metric='{metric}', Tags='{tags}'")

    def get_model_by_uid(self, uid: str) -> TorchModule:
        """
            Load a PyTorch model from its UID.
            The UID of an estimator model can be retrieved by calling self.save_estimator().
        """
        all_models = self.get_all_models()
        for logged_model in all_models:
            if logged_model.model_id == uid:
                artifact_location = logged_model.artifact_location
                model = mlflow.pytorch.load_model(artifact_location)

                return model
        else:
            raise LookupError(f"No model found matching UID: {uid}")

    @staticmethod
    def get_all_models():
        all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]

        all_models = mlflow.search_logged_models(
            experiment_ids=all_experiments,
            output_format="list"
        )

        return all_models

    def _forward_port(self):
        ssh_command = "ssh -Y -L {}:localhost:{} krepo@{}".format(
            self.server_port,
            self.server_port,
            self.server_ip,
        )
        if platform_is("Linux"):
            terminal = _get_linux_terminal()
            if terminal:
                subprocess.Popen([terminal, "-e", ssh_command])
            else:
                raise EnvironmentError("No supported terminal emulator found on this system.")
        elif platform_is("Windows"):
            subprocess.Popen(ssh_command, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif platform_is("macOS"):
            osa_command = f'''
            tell application "Terminal"
                do script "{ssh_command}"
                activate
            end tell
            '''
            subprocess.Popen(["osascript", "-e", osa_command])


def platform_is(system: str) -> bool:
    current_system = platform.system().lower()
    requested_system = system.lower()
    if current_system == requested_system:
        return True
    if requested_system == "macos" and current_system == "darwin":
        return True
    return False


# noinspection SpellCheckingInspection,PyDeprecation
def _get_linux_terminal():
    terminals = [
        "x-terminal-emulator",
        "gnome-terminal",
        "konsole",
        "xfce4-terminal",
        "lxterminal",
        "xterm",
        "terminator",
        "mate-terminal"
    ]
    for term in terminals:
        if shutil.which(term):
            return term
    return None