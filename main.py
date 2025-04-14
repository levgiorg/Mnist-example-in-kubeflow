"""
Run script for the MNIST digit recognition pipeline.

This script compiles and optionally submits the Kubeflow Pipeline.
"""

import argparse
import time
import kfp
from config.settings import DEFAULT_EXPERIMENT_NAME
from pipeline import digits_recognizer_mlflow_pipeline

def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition Pipeline')
    parser.add_argument('--compile-only', action='store_true', 
                      help='Only compile the pipeline without submitting')
    parser.add_argument('--output', type=str, default='digits_pipeline.yaml',
                      help='Output path for compiled pipeline YAML')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                      help='Optimizer to use (adam, sgd, etc.)')
    parser.add_argument('--experiment', type=str, default='digits-recognizer-kfp',
                      help='MLflow experiment name')
    parser.add_argument('--kfp-experiment', type=str, default=DEFAULT_EXPERIMENT_NAME,
                      help='Kubeflow Pipeline experiment name')
    
    args = parser.parse_args()
    
    # Compile the pipeline
    print(f"Compiling pipeline to {args.output}...")
    kfp.compiler.Compiler().compile(
        pipeline_func=digits_recognizer_mlflow_pipeline,
        package_path=args.output
    )
    print(f"Pipeline compiled successfully to {args.output}")
    
    if args.compile_only:
        print("Skipping pipeline submission (--compile-only flag set)")
        return
    
    # Run the pipeline if requested
    try:
        client = kfp.Client()
        print(f"Connected to KFP at: {client.host if hasattr(client, 'host') else 'default'}")
        
        # Create or get experiment
        try:
            experiment = client.get_experiment(experiment_name=args.kfp_experiment)
            print(f"Using existing experiment: {args.kfp_experiment}")
        except:
            experiment = client.create_experiment(args.kfp_experiment)
            print(f"Created new experiment: {args.kfp_experiment}")
        
        # Submit pipeline run
        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=f"digits-recognizer-run-{int(time.time())}",
            pipeline_package_path=args.output,
            params={
                "no_epochs": args.epochs,
                "optimizer": args.optimizer,
                "mlflow_experiment": args.experiment
            }
        )
        
        # Show run information
        print(f"Pipeline submitted with run ID: {run.id}")
        if hasattr(client, 'host') and client.host:
            print(f"View run at: {client.host}/#/runs/details/{run.id}")
    
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("\nYou can still run the pipeline manually by uploading the YAML file to the KFP UI")


if __name__ == "__main__":
    main()