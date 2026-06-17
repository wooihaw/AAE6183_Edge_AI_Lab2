import os
import time
import numpy as np
import tensorflow as tf
import openvino as ov

def optimize_and_evaluate_openvino(image_folder, export_dir="inception_v3_exported", ov_model_path="inception_v3.xml"):
    """
    Exports the Keras 3 model to a standard SavedModel directory,
    converts it via the OpenVINO directory path parser,
    and evaluates runtime performance.
    """
    # -------------------------------------------------------------------------
    # PART 1: Standard Keras 3 Export and OpenVINO Conversion
    # -------------------------------------------------------------------------
    if not os.path.exists(export_dir):
        print("Loading pretrained Inception v3 framework model...")
        native_model = tf.keras.applications.InceptionV3(weights='imagenet')
        
        print(f"Exporting model to standard SavedModel directory: {export_dir}")
        # Keras 3 requires .export() to generate a standard TensorFlow directory layout
        native_model.export(export_dir)

    print("Converting SavedModel directory to OpenVINO Intermediate Representation (IR)...")
    # Passing the directory path string avoids the in-memory object inspection bug
    ov_model = ov.convert_model(export_dir)
    
    # Serialize the resulting OpenVINO graph structure to disk (.xml and .bin files)
    ov.save_model(ov_model, ov_model_path, compress_to_fp16=True)
    print(f"OpenVINO IR model successfully serialized to: {ov_model_path}")

    # -------------------------------------------------------------------------
    # PART 2: OpenVINO Runtime Compilation and Execution
    # -------------------------------------------------------------------------
    print("Compiling model for the target Intel hardware device...")
    core = ov.Core()
    compiled_model = core.compile_model(model=ov_model, device_name="CPU")
    
    # Establish output layer reference for the compiled graph
    output_layer = compiled_model.output(0)

    # Inception v3 target resolution parameters
    target_size = (299, 299)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Error: No valid images found in directory '{image_folder}'.")
        return

    print(f"Found {len(image_files)} images. Commencing OpenVINO evaluation...")

    # Execute an engine warm-up pass using random synthetic data
    print("Performing OpenVINO runtime warm-up...")
    dummy_input = np.random.rand(1, 299, 299, 3).astype(np.float32)
    compiled_model([dummy_input])

    total_inference_time = 0.0
    processed_count = 0

    # Execute pipeline sequentially across the target image files
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        
        try:
            # Load and scale imagery using native TensorFlow utilities
            img = tf.keras.utils.load_img(image_path, target_size=target_size)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            
            # Formulate compatible input data types
            input_tensor = img_array.astype(np.float32)
            
            # Measure precise runtime inference duration
            start_time = time.perf_counter()
            _ = compiled_model([input_tensor])[output_layer]
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            total_inference_time += latency
            processed_count += 1
            
        except Exception as e:
            print(f"Skipping file {filename} due to processing error: {e}")

    # Calculate system throughput metrics post-optimization
    if processed_count > 0:
        average_latency = total_inference_time / processed_count
        fps = processed_count / total_inference_time
        
        print("\n================ OPENVINO BENCHMARK RESULTS ================")
        print(f"Total Images Processed:  {processed_count}")
        print(f"Total Inference Time:    {total_inference_time:.4f} seconds")
        print(f"Average Latency:         {average_latency:.4f} seconds per image")
        print(f"Throughput Speed (FPS):  {fps:.2f} Frames Per Second")
        print("============================================================")
    else:
        print("Evaluation aborted. No images were processed successfully.")

if __name__ == "__main__":
    # Define the absolute or relative path to the image dataset folder
    IMAGE_DIRECTORY = "test_images"
    
    # Execute optimization and evaluation pipeline
    optimize_and_evaluate_openvino(IMAGE_DIRECTORY)