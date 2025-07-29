'''
Ce fichier s occupera d appliquer le model au tenseur mit en entree

Usage: python process.py /path/to/tensor.pt [output_directory]
'''

import torch
import torchvision
import sys
import os

MODEL_PATH = "detectron2_model_1200x1200.pt"



def main(input_tenseur):

    device = torch.device("cpu")

    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()

    input_tensor = torch.load(tensor_path).to(device)  # shape: [1, 3, H, W]

    outputs = model(input_tensor[0])  # Correct
    
    return outputs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process.py /path/to/tensor.pt [output_directory]")
        sys.exit(1)

    tensor_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    output_mask = main(tensor_path)

    base_name = os.path.splitext(os.path.basename(tensor_path))[0]
    mask_path = os.path.join(output_dir, base_name + "_mask.pt")

    torch.save(output_mask, mask_path)

    print(f"Tenseur masque sauvegardé : {mask_path}")