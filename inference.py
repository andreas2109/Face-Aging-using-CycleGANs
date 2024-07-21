import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(project_root)

def load_generator(model_type, gen_subtype):
    if model_type == "base":
        from base_cyclegan.generator import Generator
    elif model_type == "attention":
        from attention_cyclegan.generator import Generator
    else:
        raise ValueError("Unknown model type")

    gen = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    checkpoint_path = os.path.join('checkpoints', f'gen{gen_subtype}.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    gen.load_state_dict(checkpoint['state_dict'])
    return gen

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image

def tensor_to_image(tensor):
    image = tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    image = (image + 1) / 2 * 255
    image = image.astype('uint8')
    return Image.fromarray(image)

def inference(input_image_path, output_image_path, model_type, gen_subtype):
    gen = load_generator(model_type, gen_subtype)
    gen.eval()

    img = load_image(input_image_path)

    # Run the inference
    with torch.no_grad():
        fake_image = gen(img)
        
    output_image = tensor_to_image(fake_image)
    output_image.save(output_image_path)
    output_image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified model and generator type")
    parser.add_argument("--model", type=str, required=True, choices=["base", "attention"], help="Model type to use for inference")
    parser.add_argument("--gen", type=str, required=True, choices=["o", "y"], help="Generator subtype to use (o or y)")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image")

    args = parser.parse_args()

    inference(args.input, args.output, args.model, args.gen)
