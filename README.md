# MNIST Digit Recognition Pipeline

A complete ML pipeline for digit recognition using Kubeflow Pipelines and MLflow integration.

## Overview

This project implements an end-to-end machine learning pipeline for MNIST digit recognition using Kubeflow Pipelines and MLflow for experiment tracking. The pipeline includes data acquisition, preprocessing, model training, and model serving, all orchestrated in a modular, maintainable codebase.

## Project Structure

```
digits_recognition_pipeline/
├── config/
│   └── settings.py           # Configuration settings
├── components/
│   ├── __init__.py
│   ├── data_acquisition.py   # Data acquisition components
│   ├── data_processing.py    # Data preprocessing components
│   ├── model_training.py     # Model training components
│   └── model_serving.py      # Model serving components
├── utils/
│   ├── __init__.py
│   ├── minio_utils.py        # MinIO helper functions
│   └── mlflow_utils.py       # MLflow helper functions
├── notebooks/
│   └── MNIST Digit Recognition.ipynb  # Demo notebook
├── pipeline.py               # Pipeline definition
├── run.py                    # Pipeline execution script
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Prerequisites

- Kubeflow Pipelines 1.8+ installed and running
- MLflow tracking server deployed
- MinIO or S3-compatible object storage
- Kubernetes cluster with KServe (for model serving)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mnist-digit-recognition-pipeline.git
   cd mnist-digit-recognition-pipeline
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment:
   - Update `config/settings.py` with your environment-specific settings
   - Alternatively, set the corresponding environment variables

## Usage

### Running the Pipeline

You can run the pipeline using the included `run.py` script:

```bash
# Run with default parameters
python run.py

# Compile only (without submitting)
python run.py --compile-only

# Customize parameters
python run.py --epochs 5 --optimizer adam --experiment my-experiment
```

### Pipeline Parameters

- `no_epochs`: Number of training epochs (default: 3)
- `optimizer`: Optimizer to use for training (default: "adam")
- `mlflow_experiment`: MLflow experiment name (default: "digits-recognizer-kfp")

### Using the Jupyter Notebook

The `notebooks/MNIST Digit Recognition.ipynb` notebook provides an interactive way to explore the pipeline:

1. Open the notebook in Jupyter or VS Code
2. Execute cells to explore data, test components, and run the pipeline
3. View results and visualizations

## Pipeline Components

### 1. Data Acquisition

Downloads the MNIST dataset and uploads it to MinIO storage.

### 2. Data Processing

Reshapes and normalizes the image data, preparing it for the CNN model.

### 3. Model Training

Builds, trains and evaluates a CNN model for digit recognition, with MLflow tracking for:
- Model parameters
- Training metrics
- Model architecture
- Confusion matrix
- Model artifacts

### 4. Model Serving

Deploys the trained model as a KServe InferenceService for online prediction.

## MLflow Integration

This pipeline leverages MLflow for:

- Experiment tracking
- Parameter logging
- Metric recording
- Model registration
- Artifact storage

The MLflow tracking UI can be accessed at the configured tracking server URL.

## Configuration

Configure the pipeline by editing `config/settings.py` or setting environment variables:

| Setting | Environment Variable | Description |
|---------|---------------------|-------------|
| MINIO_ENDPOINT | MINIO_ENDPOINT | MinIO server endpoint |
| MINIO_ACCESS_KEY | MINIO_ACCESS_KEY | MinIO access key |
| MINIO_SECRET_KEY | MINIO_SECRET_KEY | MinIO secret key |
| MLFLOW_TRACKING_URI | MLFLOW_TRACKING_URI | MLflow tracking server URI |
| K8S_NAMESPACE | K8S_NAMESPACE | Kubernetes namespace |

## Development

### Adding New Components

1. Create a new file in the `components/` directory
2. Define your component using the `@component` decorator
3. Import and use the component in `pipeline.py`

### Modifying the Pipeline

Edit `pipeline.py` to change the pipeline structure or add new components.

## Troubleshooting

### Common Issues

1. **Connection errors to MinIO**:
   - Verify MinIO endpoint and credentials in settings
   - Check network connectivity to MinIO server

2. **MLflow tracking issues**:
   - Ensure MLflow server is running
   - Verify tracking URI is correct
   - Check S3/MinIO bucket for MLflow exists

3. **KServe deployment failures**:
   - Verify KServe is properly installed
   - Check service account has correct permissions
   - Ensure model was properly saved to MinIO

## Acknowledgments

- TensorFlow and Keras for the ML framework
- Kubeflow for the pipeline orchestration
- MLflow for experiment tracking

## License

Copyright © 2025, Fourdotinfinity

## Developers

George Levis