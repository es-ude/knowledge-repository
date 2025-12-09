from krepo.API import KnowledgeRepoAPI
from model_dummy import MyTorchModule


krepo: KnowledgeRepoAPI = KnowledgeRepoAPI(
    tracking_uri="http://localhost:2035",
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
    experiment_name="Test_NAS_Experiment",
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

krepo.log_metrics({"final_accuracy": 0.9})

krepo.end_run()