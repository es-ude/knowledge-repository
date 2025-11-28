# knowledge-repository
A light-weight knowledge repository used to store neural network architecture and platform prameters.

The server can be reached via SSH Port Forwarding:

    ssh -L 2035:127.0.0.1:2035 krepo@65.108.38.237

Afterward, MLflow can be connected locally using:

    mlflow.set_tracking_uri("http://127.0.0.1:2035")

Or in a web browser via:
    
    http://127.0.0.1:2035

To gain access, an admin has to add your Public SSH Key.
