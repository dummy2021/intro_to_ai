from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    # parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    args = parser.parse_args()

    models=[]
    directory = r'checkpoints'
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            models.append(filename)
        else:
            continue

    os.makedirs("images/outputs", exist_ok=True)
    for model in models:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = style_transform()

        # Define model and load model checkpoint
        transformer = TransformerNet().to(device)
        transformer.load_state_dict(torch.load("checkpoints/"+model,'cpu'))
        # transformer.load_state_dict(torch.load("checkpoints/"+model))
        transformer.eval()

        # Prepare input
        image_tensor = Variable(transform(Image.open(args.image_path))).to(device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = denormalize(transformer(image_tensor)).cpu()

        # Save image
        fn = args.image_path.split("/")[-1]
        save_image(stylized_image, f"images/outputs/{model}-{fn}")
