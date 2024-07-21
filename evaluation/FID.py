import os
import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
from skimage.transform import resize

def load_images_from_directory(directory, target_size):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.NEAREST)
            images.append(np.array(img))
    return np.array(images)

def scale_images(images, new_shape):
    images_list = [resize(image, new_shape, 0) for image in images]
    return asarray(images_list)

#calculate FID
def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # add small value to the diagonal of covariance matrices 
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

dir1 = r'\path\images'            # add your path
dir2 = r'\path\generated_images'  # add your path

images1 = load_images_from_directory(dir1, (299, 299))
images2 = load_images_from_directory(dir2, (299, 299))

print(f'Loaded images1 shape: {images1.shape}')
print(f'Loaded images2 shape: {images2.shape}')

# check the images if the have the right shape
if images1.shape[1:] != (299, 299, 3):
    raise ValueError(f'images1 shape {images1.shape} is incorrect')
if images2.shape[1:] != (299, 299, 3):
    raise ValueError(f'images2 shape {images2.shape} is incorrect')

images1 = images1.astype('float32')
images2 = images2.astype('float32')

images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)