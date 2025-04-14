import tensorflow as tf
import numpy as np
from minio import Minio
import os
import shutil # To remove temporary directory later

print("TensorFlow Version:", tf.__version__)

# --- MinIO Configuration ---
# !!! UPDATE THESE WITH YOUR DETAILS !!!
MINIO_ENDPOINT = "localhost:9000"  # Your MinIO service IP/DNS:PORT
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "mlpipeline"
MINIO_MODEL_PREFIX = "models/detect-digits/1/" # Path to the version directory in MinIO (ensure trailing /)
LOCAL_MODEL_DIR = "/tmp/downloaded_mnist_model" # Temporary local directory

# --- Sample Data (Digit 5) ---
# (Keeping your x_number_five array as requested)
x_number_five = np.array([[[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
              18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
             253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
             253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
             253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
             205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
              90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
             190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
             253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
             241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
             148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
             253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
             253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
             195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
              11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0]]]])

# --- Function to Download Directory from MinIO ---
def download_minio_directory(client: Minio, bucket_name: str, minio_prefix: str, local_dir: str):
    """Downloads objects from MinIO prefix recursively to local directory."""
    print(f"Attempting to download model from '{bucket_name}/{minio_prefix}' to '{local_dir}'...")
    try:
        # Ensure local directory exists and is empty
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)

        objects = client.list_objects(bucket_name, prefix=minio_prefix, recursive=True)
        downloaded_count = 0
        for obj in objects:
            # Calculate local path
            relative_path = obj.object_name[len(minio_prefix):]
            local_file_path = os.path.join(local_dir, relative_path)
            local_file_dir = os.path.dirname(local_file_path)

            # Ensure local subdirectory exists
            if local_file_dir:
                os.makedirs(local_file_dir, exist_ok=True)

            # Download the file if it's not a pseudo-directory object
            if not obj.is_dir: # Check if it's an actual file
              print(f"Downloading '{obj.object_name}' to '{local_file_path}'")
              client.fget_object(bucket_name, obj.object_name, local_file_path)
              downloaded_count += 1

        if downloaded_count == 0:
             print(f"WARNING: No files found in MinIO bucket '{bucket_name}' with prefix '{minio_prefix}'")
             return False
        else:
             print(f"Downloaded {downloaded_count} files successfully.")
             return True

    except Exception as e:
        print(f"ERROR: Failed to download from MinIO: {e}")
        return False

# --- Main Execution ---
model = None
try:
    # Connect to MinIO
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False # Set to True if MinIO uses HTTPS
    )

    # Check bucket exists (optional but good practice)
    found = minio_client.bucket_exists(MINIO_BUCKET)
    if not found:
        print(f"ERROR: MinIO bucket '{MINIO_BUCKET}' not found.")
        exit()
    else:
        print(f"MinIO bucket '{MINIO_BUCKET}' found.")

    # Download the model directory
    if download_minio_directory(minio_client, MINIO_BUCKET, MINIO_MODEL_PREFIX, LOCAL_MODEL_DIR):

        # --- Load Model ---
        print(f"\n--- Loading model from local directory: {LOCAL_MODEL_DIR} ---")
        # Check if the typical SavedModel file exists
        if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "saved_model.pb")):
             print(f"ERROR: 'saved_model.pb' not found in {LOCAL_MODEL_DIR}. Check download or model structure in MinIO.")
        else:
            try:
                model = tf.keras.models.load_model(LOCAL_MODEL_DIR)
                print("Model loaded successfully.")
                model.summary() # Print model summary
            except Exception as e:
                print(f"ERROR: Failed to load model from {LOCAL_MODEL_DIR}: {e}")

except Exception as e:
    print(f"An error occurred during MinIO connection or download setup: {e}")

# --- Run Inference (only if model loaded successfully) ---
if model:
    print("\n--- Preparing data for inference ---")
    print("Original data shape:", x_number_five.shape)
    print("Actual Number: 5")

    # Normalize the data (0-255 -> 0-1) - MUST match training preprocessing
    t = x_number_five.astype(np.float32) / 255.0

    # Reshape to expected format (Batch Size, Height, Width, Channels)
    t = t.reshape(-1, 28, 28, 1)
    print("Processed data shape:", t.shape)

    # --- Perform Inference ---
    try:
        print("\n--- Running model.predict() ---")
        predictions = model.predict(t)

        # --- Interpret Response ---
        print("Raw prediction output (probabilities):", predictions)

        if predictions is not None and len(predictions) > 0:
            # Assuming the output is a list/array of probabilities for each class (0-9)
            # For a batch size of 1, predictions[0] contains the probabilities
            pred_probabilities = predictions[0]
            predicted_class = np.argmax(pred_probabilities)
            predicted_prob = pred_probabilities[predicted_class]
            print(f"\n==> Predicted Class: {predicted_class} (Probability: {predicted_prob:.4f}) <==")
        else:
            print("\nError: Model prediction returned None or empty result.")

    except Exception as e:
        print(f"An error occurred during model prediction: {e}")
else:
    print("\nExiting: Model was not loaded successfully. Cannot run inference.")