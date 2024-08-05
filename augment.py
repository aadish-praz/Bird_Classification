import Augmentor
import os

# Path to the dataset
base_path = r'/home/kusan/bird/seendataset(416)/bird dataset/train'

# List all class directories
class_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

for class_dir in class_dirs:
    class_path = os.path.join(base_path, class_dir)
    
    # Create a pipeline for the class with the output directory specified
    p = Augmentor.Pipeline(class_path, output_directory=".")
    
    p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)  # Rotate images up to 25 degrees
    p.zoom_random(probability=0.5, percentage_area=0.8)  # Random zoom
    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)  # Random brightness
    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)  # Random contrast
    p.flip_left_right(probability=0.5)  # Horizontal flip
    p.flip_top_bottom(probability=0.3)  # Vertical flip
    
    # Count the number of images in the class directory
    num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    
    # Sample a number of new images equal to the original number of images
    p.sample(num_images)

print("Augmentation completed.")
