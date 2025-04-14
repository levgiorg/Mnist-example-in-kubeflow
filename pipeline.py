"""Pipeline definition for the MNIST digit recognition workflow.

This module defines the Kubeflow Pipeline that connects all components
into a complete ML workflow.
"""

from kfp import dsl
from config.settings import PIPELINE_NAME
from components.data_acquisition import get_data_batch, get_latest_data
from components.data_processing import reshape_data
from components.model_training import model_building_with_mlflow
from components.model_serving import model_serving

@dsl.pipeline(
    name=PIPELINE_NAME,
    description='MNIST digit recognition pipeline with MLflow integration'
)
def digits_recognizer_mlflow_pipeline(
    no_epochs: int = 3,
    optimizer: str = "adam",
    mlflow_experiment: str = "digits-recognizer-kfp"
):
    """
    Define the end-to-end ML pipeline for digit recognition.
    
    This pipeline:
    1. Downloads and prepares the MNIST dataset
    2. Reshapes and normalizes the data
    3. Builds, trains and evaluates a CNN model
    4. Tracks experiments with MLflow
    5. Deploys the model with KServe
    
    Args:
        no_epochs: Number of training epochs
        optimizer: Optimizer to use for training
        mlflow_experiment: MLflow experiment name for tracking
    
    Returns:
        Dictionary with pipeline outputs including model metrics
    """
    # Step 1: Data Acquisition
    data_task = get_data_batch()
    
    # Step 2: Optional data update step
    latest_data_task = get_latest_data().after(data_task)
    
    # Step 3: Data Preprocessing
    reshape_task = reshape_data()
    reshape_task.after(data_task)
    
    # Step 4: Model Building with MLflow
    model_task = model_building_with_mlflow(
        no_epochs=no_epochs,
        optimizer=optimizer,
        experiment_name=mlflow_experiment
    )
    model_task.after(reshape_task)
    
    # Step 5: Model Serving
    serving_task = model_serving()
    serving_task.after(model_task)
    
    # Return pipeline outputs
    return {
        'model_accuracy': model_task.outputs['output_model_accuracy'],
        'model_loss': model_task.outputs['output_model_loss'],
        'mlflow_run_id': model_task.outputs['mlflow_run_id'],
        'inference_service': serving_task.output
    }