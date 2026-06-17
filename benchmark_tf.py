import os
import time
import numpy as np
import tensorflow as tf

def evaluate_native_model(image_folder):
    """
    Evaluates the native pretrained Inception v3 model on a directory of images.
    Calculates execution duration in seconds and throughput in FPS.
    """
    # Load the pretrained Inception v3 model with ImageNet weights
    print("Loading pretrained Inception v3 model...")
    model = tf.keras.applications.InceptionV3(weights='imagenet')
    
    # Inception v3 architecture requires an input resolution of 299x299
    target_size = (299, 299)
    
    # Filter directory contents for valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Error: No valid images found in directory '{image_folder}'.")
        return

    print(f"Found {len(image_files)} images. Commencing evaluation...")

    # Execute a warm-up inference to exclude framework initialization from benchmarks
    print("Performing model warm-up...")
    dummy_input = np.random.rand(1, 299, 299, 3)
    model.predict(dummy_input, verbose=0)

    total_inference_time = 0.0
    processed_count = 0

    # Iterate through the image list sequentially
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        
        try:
            # Load image and convert to the format expected by the framework
            img = tf.keras.utils.load_img(image_path, target_size=target_size)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply standard Inception v3 normalization scaling
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            
            # Precision timing of the forward pass execution
            start_time = time.perf_counter()
            model.predict(img_array, verbose=0)
            end_time = time.perf_counter()
            
            # Accumulate temporal metrics
            latency = end_time - start_time
            total_inference_time += latency
            processed_count += 1
            
        except Exception as e:
            print(f"Skipping file {filename} due to processing error: {e}")

    # Compute final throughput and execution metrics
    if processed_count > 0:
        average_latency = total_inference_time / processed_count
        fps = processed_count / total_inference_time
        
        print("\n================ BENCHMARK RESULTS ================")
        print(f"Total Images Processed: {processed_count}")
        print(f"Total Inference Time:   {total_inference_time:.4f} seconds")
        print(f"Average Latency:        {average_latency:.4f} seconds per image")
        print(f"Throughput Speed:       {fps:.2f} Frames Per Second (FPS)")
        print("===================================================")
    else:
        print("Evaluation aborted. No images were processed successfully.")

if __name__ == "__main__":
    # Define the absolute or relative path to the image dataset folder
    IMAGE_DIRECTORY = "test_images"
    
    # Execute evaluation pipeline
    evaluate_native_model(IMAGE_DIRECTORY)