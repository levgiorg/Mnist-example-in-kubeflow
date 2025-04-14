# main.py
"""
Main execution script for the MNIST digit recognition pipeline.

This script compiles the pipeline and prints the path to the YAML file.
Use this compiled YAML to submit the pipeline through the Kubeflow UI
or using the KFP client.
"""

import kfp
from pipeline import digits_recognizer_mlflow_pipeline

def main():
    """Compile the pipeline to a YAML file."""
    pipeline_filename = "mnist_pipeline.yaml"
    
    # Compile the pipeline
    print(f"Compiling pipeline to {pipeline_filename}...")
    kfp.compiler.Compiler().compile(
        pipeline_func=digits_recognizer_mlflow_pipeline,
        package_path=pipeline_filename
    )
    print(f"Pipeline compiled successfully to {pipeline_filename}")
    print(f"You can now upload this file to the Kubeflow Pipelines UI to run the pipeline.")

if __name__ == "__main__":
    main()