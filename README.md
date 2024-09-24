# image-segmentation-brain-tumor-detection

image-segmentation-brain-tumor-detection

## MLFlow

[Docs](https://mlflow.org/docs/latest/getting-started/index.html)

### Basic metrics logging during training.

Start mlflow server:

```
mlflow ui
```

```
import mlflow
# Set our tracking server uri for logging

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
# Create a new MLflow Experiment
mlflow.set_experiment("Training_Brain_Tumor")

mlflow.tensorflow.autolog(checkpoint=True, checkpoint_save_best_only=False)
```
