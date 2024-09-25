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

### Basic logging:

```
import mlflow

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
# Create a new MLflow Experiment
mlflow.set_experiment("test")

with mlflow.start_run() as run:
    # log some model parameters
    params = {"max_depth": 2, "random_state": 42}
    mlflow.log_param(params)
    # log some metrics
    mlflow.log_metric("metric1", 2)
    # log some artifacts
    mlflow.log_artifact("./test.txt")
    # log the model
    mlflow.tensorflow.log_model(model, "model")
    # log the model as a Keras model
    mlflow.keras.log_model(model, "model")
```

### Autologging:

```
mlflow.tensorflow.autolog(checkpoint=True, checkpoint_save_best_only=False)

with mlflow.start_run() as run:
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=5,
                        callbacks=[callback])
```

### Adding an MLflow Model to the Model Registry

There are three programmatic ways to add a model to the registry.

1. First, you can use the mlflow.<model_flavor>.log_model() method. For example, in your code:

```
with mlflow.start_run() as run:
    ...
    model = RandomForestRegressor(**params)
    ...
    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model",
    )
```

2. The second way is to use the mlflow.register_model() method, after all your experiment runs complete and when you have decided which model is most suitable to add to the registry. For this method, you will need the run_id as part of the runs:URI argument. Same concept as connecting an experiment to a model in the UI.

```
result = mlflow.register_model(
"runs:/d16076a3ec534311817565e6527539c0/sklearn-model", "sk-learn-random-forest-reg"
)
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
