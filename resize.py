import os
from PIL import Image

def resize_images(input_directory, output_directory, size=(416,416)):
    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Iterate over all files and subdirectories in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Construct full file path
                file_path = os.path.join(root, file)
                
                # Open an image file
                with Image.open(file_path) as img:
                    # Resize image
                    img = img.resize(size, Image.LANCZOS)
                    
                    # Construct the output file path
                    relative_path = os.path.relpath(root, input_directory)
                    output_path = os.path.join(output_directory, relative_path)
                    
                    # Create the output directory if it does not exist
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    # Save the resized image
                    img.save(os.path.join(output_path, file))

# Define input and output directories
input_directory = r'/home/kusan/bird/seendataset(416)/bird dataset/train'
output_directory = r'/home/kusan/bird/seendataset(416)/train'

# Call the function to resize images
resize_images(input_directory, output_directory)
