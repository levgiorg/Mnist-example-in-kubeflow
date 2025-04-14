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
    parser.add_argument('--output', type=str, default='mnist_pipeline.yaml',
                      help='Output path for compiled pipeline YAML')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                      help='Optimizer to use (adam, sgd, etc.)')
    parser.add_argument('--experiment', type=str, default='digits-recognizer-kfp',
                      help='MLflow experiment name')
    parser.add_argument('--kfp-experiment', type=str, default="mnist-example",
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
        
        # Create or get experiment (KFP v2 compatible)
        experiment_name = args.kfp_experiment
        try:
            # Try to get experiment first
            experiment = client.get_experiment(experiment_name=experiment_name)
            print(f"Using existing experiment: {experiment_name}")
        except:
            # Create experiment if it doesn't exist
            experiment = client.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name}")
        
        # For KFP v2, we need to extract the experiment name rather than using ID
        print(f"Submitting pipeline run to experiment: {experiment_name}")
        
        # Submit pipeline run using experiment name instead of ID for v2 compatibility
        run = client.create_run_from_pipeline_package(
            pipeline_file=args.output,
            experiment_name=experiment_name,  # Use name instead of ID
            run_name=f"mnist-run-{int(time.time())}",
            arguments={
                "no_epochs": args.epochs,
                "optimizer": args.optimizer,
                "mlflow_experiment": args.experiment
            }
        )
        
        # Show run information
        print(f"Pipeline submitted with run ID: {run.run_id}")
        if hasattr(client, 'host') and client.host:
            print(f"View run at: {client.host}/#/runs/details/{run.run_id}")
    
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("\nYou can still run the pipeline manually:")
        print(f"1. Go to the Kubeflow Pipelines UI")
        print(f"2. Create a new experiment named '{args.kfp_experiment}' if it doesn't exist")
        print(f"3. Upload the compiled pipeline YAML file: {args.output}")
        print(f"4. Start a run with your desired parameters")


if __name__ == "__main__":
    main()