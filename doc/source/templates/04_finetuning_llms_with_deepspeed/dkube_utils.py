# Copyright (c) DKube.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
from typing import Any, Dict, Optional

import mlflow

class DKubeRun:
    def __init__(self, experiment:str="finetuning", run:str="run", mlflow_tracking_uri:str="http://d3x-controller.d3x.svc.cluster.local:5000", epoch:Optional[int]=0) -> None:
        self.experiment = experiment
        self.run = run
        self._user = os.environ.get("USER", "anonymous")
        self._tags = { "mlflow.user" : self._user }
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        if mlflow_tracking_uri.startswith("https"):
            os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        mlflow.set_experiment(experiment_name=self.experiment)
        self._run_name = self.run + f"-{epoch}"
        with mlflow.start_run(run_name=self._run_name) as newrun:
            self._run_id = newrun.info.run_id

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_metrics(metrics, step)

    def log_params(self, params: Dict[str, Any]) -> None:
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_params(params)

    def log_model(self, store:str) -> None:
        with mlflow.start_run(run_id=self._run_id):
            mlflow.log_artifacts(store, artifact_path=f"finetuned-{self.run}")

            tmodel = TunedModel("finetuned-model")
            mlflow.pyfunc.log_model("finetuned-model", python_model=tmodel,
                                    artifacts={"tuned_model" : mlflow.get_artifact_uri(artifact_path=f"finetuned-{self.run}")})



class TunedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, tuned_model):
        self.tuned_model = tuned_model

    def predict(self, context, model_input):
        return {}
       

