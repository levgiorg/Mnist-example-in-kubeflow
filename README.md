# MNIST MLOps Pipeline

**Production-ready ML pipeline for digit recognition using Kubeflow Pipelines, MLflow, and KServe**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Kubeflow](https://img.shields.io/badge/Kubeflow-1.8+-orange.svg)](https://kubeflow.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-green.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

This project demonstrates a complete MLOps pipeline implementation for MNIST digit recognition, showcasing modern ML engineering practices with enterprise-grade tooling. The pipeline orchestrates data ingestion, model training, experiment tracking, and model serving in a cloud-native, scalable architecture.

### Key Features

- **🔄 End-to-end MLOps Pipeline**: Automated workflow from data acquisition to model serving
- **📊 Experiment Tracking**: Comprehensive MLflow integration for reproducible experiments
- **🎯 Model Serving**: Production-ready deployment with KServe inference service
- **📦 Containerized Components**: Cloud-native approach with Docker and Kubernetes
- **🔧 Infrastructure as Code**: Declarative pipeline definitions with Kubeflow DSL
- **🏗️ Modular Architecture**: Reusable components for scalable ML workflows

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MNIST MLOps Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│  Data Acquisition → Data Processing → Model Training → Serving  │
│       ↓                    ↓               ↓            ↓       │
│   MinIO Storage     Preprocessing     MLflow Tracking  KServe   │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
mnist-mlops-pipeline/
├── 📁 components/              # Pipeline components
│   ├── data_acquisition.py     # Data ingestion component
│   ├── data_processing.py      # Data preprocessing component
│   ├── model_training.py       # Model training with MLflow
│   └── model_serving.py        # KServe deployment component
├── 📁 config/                  # Configuration management
│   └── settings.py             # Environment-based settings
├── 📁 utils/                   # Utility modules
│   ├── minio_utils.py          # Object storage operations
│   └── mlflow_utils.py         # Experiment tracking utilities
├── 📄 pipeline.py              # Main pipeline definition
├── 📄 run.py                   # Pipeline execution script
├── 📄 requirements.txt         # Python dependencies
├── 📄 .env.example             # Environment variables template
└── 📄 README.md                # This documentation
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Kubeflow Pipelines | Workflow orchestration and pipeline management |
| **Experiment Tracking** | MLflow | Model versioning, parameter tracking, and artifact storage |
| **Model Serving** | KServe | Production model deployment and inference |
| **Object Storage** | MinIO | Data and model artifact storage |
| **ML Framework** | TensorFlow/Keras | Model development and training |
| **Infrastructure** | Kubernetes | Container orchestration and scaling |

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster with Kubeflow Pipelines installed
- MLflow tracking server deployed
- MinIO or S3-compatible storage
- Python 3.8+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mnist-mlops-pipeline.git
   cd mnist-mlops-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your environment-specific values
   ```

4. **Run the pipeline**
   ```bash
   # Execute with default parameters
   python run.py
   
   # Customize training parameters
   python run.py --epochs 10 --optimizer adam --experiment production-run
   ```

## 📊 Pipeline Components

### 1. Data Acquisition
- Downloads MNIST dataset using Keras
- Validates data integrity and structure
- Uploads processed data to MinIO storage
- Generates data quality reports

### 2. Data Processing
- Normalizes pixel values to [0,1] range
- Reshapes images for CNN input (28x28x1)
- Applies data augmentation strategies
- Splits data into train/validation/test sets

### 3. Model Training
- Builds CNN architecture with configurable parameters
- Implements early stopping and learning rate scheduling
- Tracks experiments with MLflow (parameters, metrics, artifacts)
- Generates model performance visualizations
- Saves model artifacts to MinIO

### 4. Model Serving
- Deploys trained model using KServe InferenceService
- Configures auto-scaling and resource limits
- Implements health checks and monitoring
- Provides REST API for inference requests

## 🔧 Configuration

The pipeline uses environment variables for configuration management. Copy `.env.example` to `.env` and customize:

```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Kubeflow Configuration
K8S_NAMESPACE=kubeflow
KSERVE_SA_NAME=sa-minio-kserve
```

## 📈 Experiment Tracking

The pipeline integrates deeply with MLflow for comprehensive experiment tracking:

- **Parameters**: Model architecture, training configuration, hyperparameters
- **Metrics**: Training/validation accuracy, loss, F1-score
- **Artifacts**: Model files, confusion matrices, training plots
- **Model Registry**: Versioned model management with stage transitions

Access the MLflow UI at your configured tracking server to explore experiments and compare model performance.

## 🎯 Model Serving

Models are deployed using KServe for production inference:

```bash
# Check deployment status
kubectl get inferenceservice

# Send inference request
curl -X POST http://your-kserve-endpoint/v1/models/mnist-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"image": [...]}]}'
```

## 🧪 Testing & Validation

Run the test suite to validate pipeline components:

```bash
# Run unit tests
python -m pytest tests/

# Validate pipeline compilation
python run.py --compile-only

# Run integration tests
python -m pytest tests/integration/
```

## 🔍 Monitoring & Observability

The pipeline includes comprehensive monitoring capabilities:

- **Pipeline Metrics**: Execution time, success rates, resource utilization
- **Model Performance**: Accuracy drift, prediction latency, throughput
- **Infrastructure**: Kubernetes pod health, storage utilization
- **Data Quality**: Schema validation, data drift detection

## 🛡️ Security Best Practices

- Secrets managed through Kubernetes secrets or external secret managers
- Role-based access control (RBAC) for pipeline execution
- Network policies for secure communication between components
- Container security scanning and vulnerability management

## 🚀 Deployment Strategies

### Development Environment
```bash
# Local development with Docker Compose
docker-compose up -d

# Run pipeline locally
python run.py --local
```

### Production Environment
```bash
# Deploy to Kubernetes
kubectl apply -f manifests/

# Schedule pipeline execution
python run.py --schedule "0 2 * * *"  # Daily at 2 AM
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow** team for the ML framework
- **Kubeflow** community for pipeline orchestration
- **MLflow** for experiment tracking capabilities
- **KServe** for model serving infrastructure

---

*This project demonstrates production-ready MLOps practices with modern tooling and cloud-native architecture. Perfect for showcasing end-to-end ML pipeline development skills.*