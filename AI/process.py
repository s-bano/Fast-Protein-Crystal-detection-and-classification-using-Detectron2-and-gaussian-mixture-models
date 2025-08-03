'''
Ce fichier s occupera d appliquer le model au tenseur mit en entree
Il renvoie un fichier outputs

Usage: python process.py /path/to/tensor.pt [output_directory]
'''

import numpy as np
import torchvision
import sys, os, torch
from PIL import Image


MODEL_PATH = "models/model_1200x1200_0804.pt"



def main(input_tenseur):

    device = torch.device("cpu")

    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()

    input_tensor = torch.load(tensor_path).to(device)  # shape: [1, 3, H, W]

    outputs = model(input_tensor[0])  # Correct
    print(outputs)

    # Récupérer tous les masques (N, H, W)
    boxes, labels, masks, scores, image_size = outputs
    masks = masks.cpu().numpy()  # (N, H, W)

    # Combiner tous les masques en un seul
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255  # (H, W), 0 ou 255

    # Sauvegarder l'image finale
    Image.fromarray(combined_mask).save("masque_combine.png")
    
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