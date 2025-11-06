from krepo.API import KnowledgeRepoAPI

krepo: KnowledgeRepoAPI = KnowledgeRepoAPI(
    tracking_uri="http://localhost:2035",
)
krepo.set_experiment("Test_NAS_Experiment")

nas_conf = {
    "optimizer": "adam",
    "layers": 5,
}

search_conf = {
    "type": "grid",
    "range": [1, 10],
}
run_tags = [
    {
        "key": "hw_platform",
        "value": "pico",
    },
]

# Start run needs the configs
krepo.start_run(
    nas_config=nas_conf,
    search_space_config=search_conf,
    tags=run_tags,
    run_name="test run2"
)

# Log intermediate
krepo.log_metrics({"accuracy": 0.8}, step=1)
krepo.log_metrics({"accuracy": 0.85}, step=2)

# Log final
model_architecture = {
    "1_layer" : {
        "linear": {
            "width": 32,
            "activation_function": "nn.Sigmoid"
            }
        }
    }
krepo.log_model_architecture("CNN", model_architecture)
krepo.log_metrics({"final_accuracy": 0.9})

krepo.end_run()