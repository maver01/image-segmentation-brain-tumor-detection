# image-segmentation-brain-tumor-detection

Disclaimer: written with AI tools.

image-segmentation-brain-tumor-detection

## MLFlow

[Docs](https://mlflow.org/docs/latest/getting-started/index.html)

### Basic metrics logging during training.

Start mlflow server:

```
mlflow server --host 127.0.0.1 --port 8080
```

or

```
mlflow ui
```

Basic logging:

```
import mlflow

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
# Create a new MLflow Experiment
mlflow.set_experiment("test")

with mlflow.start_run() as run:
    # log some dummy variables
    mlflow.log_param("param1", 1)
    # log some metrics
    mlflow.log_metric("metric1", 2)
    # log some artifacts
    mlflow.log_artifact("./test.txt")
    # log the model
    mlflow.tensorflow.log_model(model, "model")
    # log the model as a Keras model
    mlflow.keras.log_model(model, "model")
```

Autologging:

```
mlflow.tensorflow.autolog(checkpoint=True, checkpoint_save_best_only=False)

with mlflow.start_run() as run:
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=5,
                        callbacks=[callback])
```

### Pyfunc

1.  Define a Custom Model Class.

    You need to define a Python class
    that implements `mlflow.pyfunc.PythonModel`. This class should have:

    - `predict(context, model_input)`: This method is called for inference. model_input is typically a pandas DataFrame.
    - `load_context(context)`: This method loads your model into memory.

    ```
    import mlflow.pyfunc
    import pandas as pd

    class MyCustomModel(mlflow.pyfunc.PythonModel):

        def load_context(self, context):
        # Load model (e.g., from file, or any other required setup)
        self.model = ... # Your model loading logic here

        def predict(self, context, model_input):
        # Apply the model for prediction
        return self.model.predict(model_input)
    ```

2.  Save the Model

    Once your model class is defined, you can save it using mlflow.

    ```
    mlflow.pyfunc.save_model(
        path="my_model",
        python_model=MyCustomModel()
    )
    ```

3.  Load and Use the Model

    After saving the model, you can load it using mlflow.pyfunc.load_model() and use it for prediction.

    ```
    loaded_model = mlflow.pyfunc.load_model("my_model")
    # Example inference
    data = pd.DataFrame(...)  # Input data
    predictions = loaded_model.predict(data)

    ```

4.  Log Model to MLFlow Tracking

    If you want to log your model to an MLFlow tracking server for versioning or deployment, you can use `mlflow.pyfunc.log_model()`.

    ```
    with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=MyCustomModel()
    )
    ```

5.  Server the Model

    After logging the model, you can serve it using MLFlow’s serving capabilities.

    ```
    mlflow models serve -m "runs:/<run-id>/model"
    ```
