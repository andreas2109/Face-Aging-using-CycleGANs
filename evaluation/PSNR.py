import os
from math import log10, sqrt
import cv2
import numpy as np


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def calculate_psnr_for_folders(real_images_folder, generated_images_folder):
    real_images = sorted(os.listdir(real_images_folder))
    generated_images = sorted(os.listdir(generated_images_folder))
    
    if len(real_images) != len(generated_images):
        print("The number of images in the folders do not match.")
        return
    
    for real_image_name, generated_image_name in zip(real_images, generated_images):
        real_image_path = os.path.join(real_images_folder, real_image_name)
        generated_image_path = os.path.join(generated_images_folder, generated_image_name)
        
        real_image = cv2.imread(real_image_path)
        generated_image = cv2.imread(generated_image_path)
        
        if real_image is None or generated_image is None:
            print(f"Error reading images {real_image_name} or {generated_image_name}")
            continue
        
        real_image_size = (real_image.shape[1], real_image.shape[0])
        generated_image_resized = resize_image(generated_image, real_image_size)
        
        psnr_value = PSNR(real_image, generated_image_resized)
        print(f"PSNR between {real_image_name} and {generated_image_name}: {psnr_value} dB")


real_images_folder = '\path\real_images'
generated_images_folder = '\path\generated_images'
calculate_psnr_for_folders(real_images_folder, generated_images_folder)
