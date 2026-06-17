import os
import time
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort

def convert_and_evaluate_onnx(image_folder, onnx_model_path="inception_v3.onnx"):
    """
    Converts a native Keras model directly to ONNX format using the Python API
    and evaluates inference performance using ONNX Runtime.
    """
    # -------------------------------------------------------------------------
    # PART 1: Programmatic In-Memory Conversion to ONNX Artifact
    # -------------------------------------------------------------------------
    if not os.path.exists(onnx_model_path):
        print("Loading pretrained Inception v3 framework model...")
        native_model = tf.keras.applications.InceptionV3(weights='imagenet')
        
        print("Converting Keras model directly to ONNX format via Python API...")
        # Specify the concrete tensor input specification for Inception v3
        input_spec = [tf.TensorSpec([None, 299, 299, 3], tf.float32, name="input")]
        
        # Execute the in-memory conversion pipeline
        model_proto, _ = tf2onnx.convert.from_keras(
            native_model, 
            input_signature=input_spec, 
            opset=15
        )
        
        # Serialize the generated protocol buffer directly to disk
        with open(onnx_model_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"ONNX model binary successfully serialized to: {onnx_model_path}")
    else:
        print(f"Existing ONNX model found at '{onnx_model_path}'. Skipping conversion.")

    # -------------------------------------------------------------------------
    # PART 2: ONNX Runtime Session Configuration and Benchmarking
    # -------------------------------------------------------------------------
    print("Initializing ONNX Runtime inference session on host CPU...")
    # Initialize the inference session targeting the standard CPU backend
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # Extract structural input and output metadata from the compiled graph
    input_node = session.get_inputs()[0]
    input_name = input_node.name
    output_name = session.get_outputs()[0].name
    
    # Inception v3 target resolution parameters
    target_size = (299, 299)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Error: No valid images found in directory '{image_folder}'.")
        return

    print(f"Found {len(image_files)} images. Commencing ONNX evaluation...")

    # Execute an engine warm-up pass using random synthetic data
    print("Performing ONNX Runtime execution warm-up...")
    dummy_input = np.random.rand(1, 299, 299, 3).astype(np.float32)
    session.run([output_name], {input_name: dummy_input})

    total_inference_time = 0.0
    processed_count = 0

    # Process target files sequentially to gather performance metrics
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        
        try:
            # Load and normalize imagery using framework preprocessing utilities
            img = tf.keras.utils.load_img(image_path, target_size=target_size)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            
            # Cast dataset array explicitly to match the required ONNX format
            input_tensor = img_array.astype(np.float32)
            
            # Record execution latency
            start_time = time.perf_counter()
            _ = session.run([output_name], {input_name: input_tensor})
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            total_inference_time += latency
            processed_count += 1
            
        except Exception as e:
            print(f"Skipping file {filename} due to processing error: {e}")

    # Compute systemic throughput data
    if processed_count > 0:
        average_latency = total_inference_time / processed_count
        fps = processed_count / total_inference_time
        
        print("\n================ ONNX BENCHMARK RESULTS ================")
        print(f"Total Images Processed:  {processed_count}")
        print(f"Total Inference Time:    {total_inference_time:.4f} seconds")
        print(f"Average Latency:         {average_latency:.4f} seconds per image")
        print(f"Throughput Speed (FPS):  {fps:.2f} Frames Per Second")
        print("========================================================")
    else:
        print("Evaluation aborted. No images were processed successfully.")

if __name__ == "__main__":
    # Define the absolute or relative path to the image dataset folder
    IMAGE_DIRECTORY = "test_images"
    
    # Execute conversion and benchmarking loop
    convert_and_evaluate_onnx(IMAGE_DIRECTORY)