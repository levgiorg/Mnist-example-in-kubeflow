"""Model training component for the digit recognition pipeline.

This module defines the component that builds, trains, and evaluates 
the digit recognition model with MLflow integration.
"""

from kfp.dsl import component, Artifact, Metrics
from typing import NamedTuple
from config.settings import TF_BASE_IMAGE

@component(
    base_image=TF_BASE_IMAGE,
    packages_to_install=['mlflow>=2.0.0', 'boto3']
)
def model_building_with_mlflow(
    no_epochs: int,
    optimizer: str,
    experiment_name: str = "digits-recognizer",
) -> NamedTuple(
    "Outputs",
    [
        ("model_view", Artifact),
        ("model_metrics", Metrics),
        ("output_model_accuracy", float),
        ("output_model_loss", float),
        ("mlflow_run_id", str)
    ]
):
    """
    Build, train, and evaluate a CNN model for digit recognition with MLflow tracking.
    
    This component:
    - Creates a CNN model using TensorFlow/Keras
    - Logs model architecture, parameters, and training config to MLflow
    - Trains the model on the preprocessed MNIST data
    - Evaluates model performance on test data
    - Generates confusion matrix and performance visualizations
    - Saves model artifacts to MinIO for deployment
    - Registers the model in MLflow Model Registry
    
    Args:
        no_epochs: Number of training epochs
        optimizer: Optimizer to use (e.g., 'adam', 'sgd')
        experiment_name: MLflow experiment name for tracking
        
    Returns:
        NamedTuple containing model visualization, metrics, accuracy, loss, 
        and MLflow run ID.
    """
    import json
    import os
    import shutil
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from minio import Minio
    import mlflow
    import mlflow.tensorflow
    import boto3

    print(f"Building model with: epochs={no_epochs}, optimizer={optimizer}")
    
    # ===== MLflow Setup =====
    # Configure environment variables
    os.environ["MLFLOW_TRACKING_URI"] = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://mlflow-server.kubeflow.svc.cluster.local:5000"
    )
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get(
        "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
    )
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", "your-secret-key"
    )
    
    # Ensure MLflow bucket exists
    default_bucket_name = "mlflow"
    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=boto3.session.Config(signature_version="s3v4"),
    )
    
    try:
        buckets_response = s3_client.list_buckets()
        existing_buckets = [bucket["Name"] for bucket in buckets_response["Buckets"]]
        
        if default_bucket_name not in existing_buckets:
            print(f"Creating MLflow bucket: {default_bucket_name}")
            s3_client.create_bucket(Bucket=default_bucket_name)
    except Exception as e:
        print(f"Warning: MLflow bucket check failed: {e}")
    
    # Create or get MLflow experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"MLflow experiment setup issue: {e}")
        experiment_id = None
    
    # ===== Data Loading Setup =====
    minio_client = Minio(
        "localhost:9000",
        access_key="minio",
        secret_key="your-secret-key",
        secure=False
    )
    minio_bucket = "mlpipeline"
    local_tmp_dir = "/tmp"
    
    # Define storage paths
    storage_paths = {
        "x_train": "x_train",
        "y_train": "y_train", 
        "x_test": "x_test",
        "y_test": "y_test"
    }
    
    local_paths = {k: os.path.join(local_tmp_dir, v + ".npy") for k, v in storage_paths.items()}
    model_save_path = os.path.join(local_tmp_dir, "detect-digits")
    model_upload_prefix = "models/detect-digits/1"
    
    # ===== Model Definition =====
    print("Creating CNN model for digit recognition...")
    model = keras.models.Sequential([
        # Convolutional layer
        keras.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        
        # Flatten for dense layers
        keras.layers.Flatten(),
        
        # Dense layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        
        # Output layer (10 classes for digits 0-9)
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Capture model summary
    model_summary_lines = []
    model.summary(print_fn=lambda x: model_summary_lines.append(x))
    model_summary_text = "\n".join(model_summary_lines)
    print(model_summary_text)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    
    # ===== Load Training Data =====
    print("Loading training data from MinIO...")
    try:
        # Load training data
        minio_client.fget_object(minio_bucket, storage_paths["x_train"], local_paths["x_train"])
        x_train = np.load(local_paths["x_train"])
        
        minio_client.fget_object(minio_bucket, storage_paths["y_train"], local_paths["y_train"])
        y_train = np.load(local_paths["y_train"])
        
        print(f"Loaded x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        raise
    
    # ===== Start MLflow Run =====
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow_run_id = run.info.run_id
        print(f"Started MLflow run: {mlflow_run_id}")
        
        # Log model parameters
        mlflow.log_params({
            "epochs": no_epochs,
            "optimizer": optimizer,
            "model_type": "CNN",
            "input_shape": str((28, 28, 1)),
            "output_classes": 10,
            "conv_filters": 64,
            "dense_units": "64,32,10",
            "batch_size": 20,
            "activation": "relu,relu,softmax"
        })
        
        # Log model architecture
        mlflow.log_text(model_summary_text, "model_architecture.txt")
        
        # ===== Train Model =====
        print(f"Training model for {no_epochs} epochs...")
        try:
            history = model.fit(
                x=x_train,
                y=y_train,
                epochs=no_epochs,
                batch_size=20,
                verbose=2  # Show progress for each epoch
            )
            print("Training completed successfully")
            
            # Log training history
            for epoch, (loss, acc) in enumerate(zip(
                history.history['loss'], 
                history.history['accuracy']
            )):
                mlflow.log_metrics({
                    "train_loss": loss,
                    "train_accuracy": acc
                }, step=epoch)
                
        except Exception as e:
            print(f"Error during model training: {e}")
            raise
        
        # ===== Load Test Data for Evaluation =====
        print("Loading test data for evaluation...")
        try:
            # Load test data
            minio_client.fget_object(minio_bucket, storage_paths["x_test"], local_paths["x_test"])
            x_test = np.load(local_paths["x_test"])
            
            minio_client.fget_object(minio_bucket, storage_paths["y_test"], local_paths["y_test"])
            y_test = np.load(local_paths["y_test"])
            
            print(f"Loaded x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
        except Exception as e:
            print(f"Error loading test data: {e}")
            raise
        
        # ===== Evaluate Model =====
        print("Evaluating model on test data...")
        test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
        
        # Log test metrics to MLflow
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss
        })
        
        # ===== Generate Confusion Matrix =====
        print("Generating confusion matrix...")
        test_predictions = model.predict(x=x_test)
        predicted_classes = np.argmax(test_predictions, axis=1)
        
        confusion_matrix = tf.math.confusion_matrix(
            labels=y_test, 
            predictions=predicted_classes
        ).numpy()
        
        # Format confusion matrix for visualization
        labels = list(map(str, range(10)))  # 0-9 digits
        confusion_data = []
        
        for true_idx, row in enumerate(confusion_matrix):
            for pred_idx, count in enumerate(row):
                confusion_data.append((labels[true_idx], labels[pred_idx], int(count)))
        
        df_confusion = pd.DataFrame(
            confusion_data, 
            columns=['target', 'predicted', 'count']
        )
        
        # Save confusion matrix for MLflow
        cm_file_path = os.path.join(local_tmp_dir, "confusion_matrix.csv")
        df_confusion.to_csv(cm_file_path, index=False)
        mlflow.log_artifact(cm_file_path, "confusion_matrix")
        
        # Format confusion matrix for KFP UI
        cm_csv = df_confusion.to_csv(header=False, index=False)
        
        # ===== Save Model =====
        print(f"Saving model to {model_save_path}...")
        if os.path.exists(model_save_path):
            shutil.rmtree(model_save_path)
            
        os.makedirs(model_save_path)
        keras.models.save_model(model, model_save_path)
        
        # Log model to MLflow
        mlflow.tensorflow.log_model(
            model, 
            artifact_path="model",
            registered_model_name="digits-recognizer-model"
        )
        print("Model saved and logged to MLflow")
        
        # ===== Upload Model to MinIO =====
        print(f"Uploading model to MinIO: {minio_bucket}/{model_upload_prefix}")
        
        # Helper function to upload directory recursively
        def upload_directory_to_minio(local_dir, bucket, prefix):
            """Upload directory contents recursively to MinIO"""
            for root, _, files in os.walk(local_dir):
                for file in files:
                    # Get full path
                    local_file_path = os.path.join(root, file)
                    
                    # Get relative path from the source directory
                    rel_path = os.path.relpath(local_file_path, local_dir)
                    
                    # Construct MinIO object path with normalized separators
                    minio_key = f"{prefix}/{rel_path}".replace("\\", "/")
                    
                    print(f"  Uploading: {local_file_path} → {bucket}/{minio_key}")
                    try:
                        minio_client.fput_object(bucket, minio_key, local_file_path)
                    except Exception as e:
                        print(f"  Error uploading {local_file_path}: {e}")
                        raise
        
        try:
            upload_directory_to_minio(model_save_path, minio_bucket, model_upload_prefix)
            print("Model successfully uploaded to MinIO")
        except Exception as e:
            print(f"Error uploading model to MinIO: {e}")
            raise
        
        # ===== Prepare KFP UI Outputs =====
        
        # Combined view: confusion matrix + markdown report
        model_view_json = {
            "outputs": [
                # Confusion matrix visualization
                {
                    "type": "confusion_matrix",
                    "format": "csv",
                    "schema": [
                        {'name': 'target', 'type': 'CATEGORY'},
                        {'name': 'predicted', 'type': 'CATEGORY'},
                        {'name': 'count', 'type': 'NUMBER'},
                    ],
                    "source": cm_csv,
                    "storage": "inline",
                    "labels": labels
                },
                # Markdown report
                {
                    'storage': 'inline',
                    'source': f'''# Model Training & Evaluation Report

## Training Configuration
* **Epochs:** {no_epochs}
* **Optimizer:** {optimizer}
* **Batch Size:** 20
* **Loss Function:** sparse_categorical_crossentropy

## MLflow Tracking
* **Experiment:** {experiment_name}
* **Run ID:** {mlflow_run_id}
* **Tracking Server:** {os.environ["MLFLOW_TRACKING_URI"]}

## Model Architecture
```
{model_summary_text}
```

## Evaluation on Test Set
* **Accuracy:** {test_accuracy:.4f}
* **Loss:** {test_loss:.4f}

## Model Storage
* **MinIO Path:** {minio_bucket}/{model_upload_prefix}
* **Registered in MLflow:** Yes (as "digits-recognizer-model")
''',
                    'type': 'markdown',
                }
            ]
        }
        
        # Metrics for KFP UI
        metrics_json = {
            'metrics': [
                {
                    'name': 'model-accuracy',
                    'numberValue': float(test_accuracy),
                    'format': "PERCENTAGE"
                },
                {
                    'name': 'model-loss',
                    'numberValue': float(test_loss),
                    'format': "RAW"
                }
            ]
        }
    
    # ===== Clean Up =====
    print("Cleaning up temporary files...")
    for path in local_paths.values():
        if os.path.exists(path):
            os.remove(path)
            
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
        
    if os.path.exists(cm_file_path):
        os.remove(cm_file_path)
    
    return (
        json.dumps(model_view_json),
        json.dumps(metrics_json),
        float(test_accuracy),
        float(test_loss),
        mlflow_run_id
    )