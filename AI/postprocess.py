'''
outputs = predictor(img)
outputs = filter_false_positives(img, outputs, score_thresh=0.80, area_thresh_ratio=0.3, black_thresh_ratio=0.5)
'''

import numpy as np
import cv2


def filter_false_positives(img, outputs, score_thresh=0.80, area_thresh_ratio=0.3, black_thresh_ratio=0.5):
    """
    img: image BGR (OpenCV)
    outputs: output Detectron2 predictor
    """

    # 1️⃣ Calculer le % de pixels noirs dans l'image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < 10)  # seuil pour considérer "noir"
    total_pixels = gray.shape[0] * gray.shape[1]
    black_ratio = black_pixels / total_pixels

    if black_ratio < black_thresh_ratio:
        # Pas besoin de filtrer
        return outputs

    # 2️⃣ Filtrage des instances
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()

    img_area = img.shape[0] * img.shape[1]
    keep_indices = []

    for i in range(len(instances)):
        score = scores[i]
        box = boxes[i]
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)

        if score < score_thresh and area > area_thresh_ratio * img_area:
            # On skip cette détection car faux positif probable
            continue
        else:
            keep_indices.append(i)

    # Créer les instances filtrées
    filtered_instances = instances[keep_indices]

    # Retourner un outputs modifié
    outputs["instances"] = filtered_instances
    return outputs


def reverse_image():
    return