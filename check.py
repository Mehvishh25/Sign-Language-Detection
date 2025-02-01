import cv2
import os

# Path to your original dataset folder
input_folder = 'asl-dataset/asl_dataset'  # Adjust this path to your dataset
output_folder = 'resized_images'  # Folder where resized images will be saved

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define target image size
target_size = (300, 300)

# Iterate over each folder (0-9)
for label in os.listdir(input_folder):
    label_path = os.path.join(input_folder, label)  # Path for each label folder (0-9)
    resized_label_path = os.path.join(output_folder, label)  # Path for resized label folder (0-9)

    # Create the folder if it doesn't exist
    if not os.path.exists(resized_label_path):
        os.makedirs(resized_label_path)

    # Iterate through all images in the label folder
    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)  # Full path to the image
        img = cv2.imread(image_path)  # Read image

        # Check if the image was read successfully
        if img is not None:
            # Resize the image
            resized_img = cv2.resize(img, target_size)

            # Save the resized image to the new folder
            resized_image_path = os.path.join(resized_label_path, image_name)
            cv2.imwrite(resized_image_path, resized_img)

            print(f"Resized and saved {image_name} to {resized_label_path}")

print("Image resizing complete!")
