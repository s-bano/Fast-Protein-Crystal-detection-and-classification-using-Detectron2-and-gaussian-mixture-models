import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt


def resize_and_pad(img: np.ndarray, target_size: int = 1200) -> np.ndarray:
    """
    Resize an image keeping aspect ratio, then pad it to (target_size x target_size).
    
    Args:
        img (np.ndarray): Input image (H x W x C).
        target_size (int): Desired output size (square).
    
    Returns:
        np.ndarray: Output image (target_size x target_size x C).
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a black background image
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Compute top-left corner for centering
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # Place the resized image onto the black background
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return padded_img



def preprocess_image_to_tensor(img, output_path: str = "image.pt"):
    """
    Charge une image, la resize/pad à `size x size`, la convertit en Tensor normalisé et la sauvegarde en `.pt`.

    Args:
        image_path (str): Chemin vers l'image.
        output_path (str): Chemin du fichier .pt de sortie.
        size (int): Taille de sortie (image carrée).
    
    Returns:
        torch.Tensor: Le tensor image preprocessé (C x H x W).
    """

    # Transforms TorchVision
    transform = T.Compose([
        T.ToTensor(),  # Convertit à float32 et met en CxHxW, [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406],  # Normalisation ImageNet standard
                    std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img)  # C x H x W

    # Sauvegarde au format .pt
    torch.save(img_tensor, output_path)
    return img_tensor


image_path = 'image_0001.jpg'
img = cv2.imread(image_path)
padded_img = resize_and_pad(img)

img_tensor = preprocess_image_to_tensor(padded_img, 'test_tensor.pt')