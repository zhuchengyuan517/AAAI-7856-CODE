# Import necessary libraries
import os
import cv2
import numpy as np

# Define function to augment dataset
def augment_dataset(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each image in the input directory
    for filename in os.listdir(input_dir):
        try:
            # Read image
            img = cv2.imdecode(np.fromfile(os.path.join(input_dir, filename), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Apply horizontal flip
            img_flip = cv2.flip(img, 1)

            # Apply rotation
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
            img_rotate = cv2.warpAffine(img, M, (cols, rows))

            # Apply scaling
            img_scale = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

            # Apply Gaussian blur
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)

            # Apply vertical flip
            img_flip_v = cv2.flip(img, 0)

            # Save augmented images to output directory
            cv2.imencode('.png', img_flip)[1].tofile(os.path.join(output_dir, filename[:-4] + '_flip.png'))
            cv2.imencode('.png', img_scale)[1].tofile(os.path.join(output_dir, filename[:-4] + '_scale.png'))
            cv2.imencode('.png', img_blur)[1].tofile(os.path.join(output_dir, filename[:-4] + '_blur.png'))
        except:
            print("can't open/read file: ", filename)

# Set input and output directories
input_dir = 'D:/data/Alarm area'
output_dir = 'D:/data/Alarm area-aug'

# Call function to augment dataset
augment_dataset(input_dir, output_dir)