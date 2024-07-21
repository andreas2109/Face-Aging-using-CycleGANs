import os
import cv2
from deepface import DeepFace

folder_path_gen = r'\path\generated_images' 
def estimate_age_for_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            image_path = os.path.join(directory, filename)
            
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Error: Unable to load image {filename}")
                continue
            
            try:
                result = DeepFace.analyze(img, actions=['age'])
                predicted_age = result[0]['age']  #
                print(f"Predicted Age for {filename}: {predicted_age}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

estimate_age_for_images(folder_path_gen)
