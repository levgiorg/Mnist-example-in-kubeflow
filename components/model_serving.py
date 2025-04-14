# components/model_serving.py
"""Model serving component for the digit recognition pipeline.

This module defines the component that creates a KServe
InferenceService for the trained model.
"""

from kfp.dsl import component
from config.settings import TF_BASE_IMAGE

@component(
    base_image=TF_BASE_IMAGE,
    packages_to_install=['kserve==0.8.0.1']
)
def model_serving() -> str:
    """
    Create a KServe InferenceService for the trained model.
    
    This component:
    - Configures Kubernetes client for the current environment
    - Creates a KServe InferenceService definition with appropriate settings
    - Deploys the service to make the model available for inference
    
    Returns:
        The name of the created KServe InferenceService
    """
    import os
    from datetime import datetime
    from kubernetes import client, config
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TFServingSpec

    print("Setting up model serving with KServe...")
    
    # ===== Kubernetes Configuration =====
    try:
        # Try in-cluster config first (when running in Kubeflow)
        config.load_incluster_config()
        print("Using in-cluster Kubernetes configuration")
    except config.ConfigException:
        try:
            # Fall back to local kubeconfig
            config.load_kube_config()
            print("Using local Kubernetes configuration (kubeconfig)")
        except config.ConfigException as e:
            print(f"ERROR: Could not configure Kubernetes client: {e}")
            raise
    
    # ===== Determine Namespace =====
    namespace = utils.get_default_target_namespace()
    if not namespace:
        # Try to get namespace from service account
        sa_namespace_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
        try:
            with open(sa_namespace_path, 'r') as f:
                namespace = f.read().strip()
            print(f"Using namespace from service account: {namespace}")
        except FileNotFoundError:
            namespace = 'kubeflow'  # Default namespace
            print(f"Using default namespace: {namespace}")
    
    # ===== Service Configuration =====
    minio_bucket = os.environ.get('MINIO_BUCKET', 'mlpipeline')
    storage_uri = f"s3://{minio_bucket}/models/detect-digits"
    service_account = os.environ.get('KSERVE_SA_NAME', "sa-minio-kserve")
    
    # Generate unique service name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    service_name = f'digits-recognizer-{timestamp}'
    
    kserve_version = 'v1beta1'
    api_version = f"{constants.KSERVE_GROUP}/{kserve_version}"
    
    print(f"Creating KServe InferenceService '{service_name}'")
    print(f"Model storage URI: {storage_uri}")
    print(f"Service account: {service_account}")
    print(f"Namespace: {namespace}")
    
    # ===== Create InferenceService Definition =====
    isvc = V1beta1InferenceService(
        api_version=api_version,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=service_name,
            namespace=namespace,
            annotations={"sidecar.istio.io/inject": "false"}
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name=service_account,
                tensorflow=V1beta1TFServingSpec(
                    storage_uri=storage_uri,
                )
            )
        )
    )
    
    # ===== Deploy Service =====
    kserve_client = KServeClient()
    try:
        print("Creating KServe InferenceService...")
        kserve_client.create(isvc, namespace=namespace)
        print(f"Successfully created InferenceService: {service_name}")
    except Exception as e:
        print(f"ERROR creating InferenceService: {e}")
        try:
            # Try to get status of potentially partially created service
            status = kserve_client.get(service_name, namespace=namespace)
            print(f"Current status: {status}")
        except Exception as get_e:
            print(f"Could not get status: {get_e}")
        raise
    
    return service_name