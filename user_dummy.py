from krepo.API import KnowledgeRepoAPI
import torch.nn as nn
import mlflow

class MyTorchModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyTorchModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

krepo: KnowledgeRepoAPI = KnowledgeRepoAPI(
    server_ip = "65.108.38.237",
    server_port = 2035,
)

nas_conf = {
    "optimizer": "adam",
    "layers": 5,
}
search_conf = {
    "type": "grid",
    "range": [1, 10],
}

krepo.set_experiment(
    experiment_name="Test_NAS_Experiment4",
    nas_config=nas_conf,
    search_space_config=search_conf,
    hw_platform="pico"
)
krepo.start_run(
    run_name="test run"
)
krepo.log_metrics({"accuracy": 0.8}, step=1)
krepo.log_metrics({"accuracy": 0.85}, step=2)
krepo.log_metrics({"accuracy": 0.87}, step=4)

model = MyTorchModule(10, 20, 5)
model_params = {
    "task_type": "Regression",
    "model_family": "Linear",
    "data_domain": "Timeseries"
    }
model_architecture = {
    "1_layer" : {
        "linear": {
            "width": 32,
            "activation_function": "nn.Sigmoid"
            }
        }
    }
krepo.log_model(model_params, model_architecture, model)

krepo.end_run()

training_data = krepo.get_training_data_for_estimator("accuracy", "pico")
print(training_data)
krepo.start_run("run 2")
krepo.save_estimator(estimator=model, hw_platform="pico", metric="accuracy", validation_loss=0.3)
krepo.end_run()
print("test1")
estimator = krepo.load_estimator("pico", "accuracy")
print("test2")
