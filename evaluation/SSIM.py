import os
import math
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range
    pad = window_size // 2
    _, channels, height, width = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channels).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

def main(real_images_folder, generated_images_folder):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    real_images = sorted([os.path.join(real_images_folder, f) for f in os.listdir(real_images_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    generated_images = sorted([os.path.join(generated_images_folder, f) for f in os.listdir(generated_images_folder) if f.endswith(('png', 'jpg', 'jpeg'))])

    if len(real_images) != len(generated_images):
        print("The number of real and generated images must be the same.")
        return

    ssim_scores = []
    for real_img_path, gen_img_path in zip(real_images, generated_images):
        real_img = load_image(real_img_path, transform)
        gen_img = load_image(gen_img_path, transform)

        ssim_score = ssim(real_img, gen_img, val_range=255)
        ssim_scores.append(ssim_score.item())
        print(f"SSIM score for {os.path.basename(real_img_path)} and {os.path.basename(gen_img_path)}: {ssim_score.item()}")

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    print(f"Average SSIM score: {avg_ssim}")

if __name__ == "__main__":


    real_images_folder = '\path\real_images'
    generated_images_folder = '\path\generated_images'
    main(real_images_folder, generated_images_folder)