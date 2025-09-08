import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import sahi
from sahi import AutoDetectionModel
from sahi.predict import predict, get_sliced_prediction, get_prediction
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.utils.cv import read_image
import seaborn as sns
import os
import shutil
import glob



DETECT_MODEL = '/Users/pagatok/Projets/Stage/crystaldetection/models/best8-detect.pt'
SEGMENT_MODEL = '/Users/pagatok/Projets/Stage/crystaldetection/models/seg-final-yolo.pt'


'''
Contains the Pipeline class definition, along with utility functions for result merging
the functions defined below are just utility functions meant to be used to calculate the two step Non max suppression process.
First, we run the image through a SAHI auto detection model and seperately through a core YOLOv12 
'''

def compute_intersection(boxA, boxB):
    '''
    computes intersection area between two bounding boxes, using their coordinates
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def asymmetric_nms(
    yolo_boxes,
    yolo_scores,
    sahi_boxes,
    sahi_scores,
    overlap_thresh=0.4,
):
    '''
    Asymmetric NMS here is used to suppress bounding boxes from SAHI using the bounding boxes from a raw
    YOLOv12 detection. The process keeps all the yolo boxes, but filters out the sahi boxes having a 
    significant overlap with them. the metric used here, instead of iou is ios, the intersection over
    area of the smaller object. This is done to ignore small parts of a crystal and replace it with the actual 
    bigger crystal bounding box.
    '''
    yolo_boxes = np.array(yolo_boxes, dtype=float)
    yolo_scores = np.array(yolo_scores, dtype=float)
    sahi_boxes = np.array(sahi_boxes, dtype=float)
    sahi_scores = np.array(sahi_scores, dtype=float)

    keep_sahi = np.ones(len(sahi_boxes), dtype=bool)

    for i, s_box in enumerate(sahi_boxes):
        for y_box in yolo_boxes:
            inter = compute_intersection(s_box, y_box)
            if inter == 0:
                continue
            ratio = inter / min(box_area(s_box), box_area(y_box))
            if ratio > overlap_thresh:
                keep_sahi[i] = False
                break

    final_boxes = list(yolo_boxes) + [b for k, b in zip(keep_sahi, sahi_boxes) if k]
    final_scores = list(yolo_scores) + [s for k, s in zip(keep_sahi, sahi_scores) if k]
    final_sources = (
        [0] * len(yolo_boxes) + [1 for k in keep_sahi if k]
    )  # 0 = YOLO, 1 = SAHI
    return np.array(final_boxes), np.array(final_scores), np.array(final_sources)

def fuse_into_sahi(
    image_path,
    sahi_result: PredictionResult,
    yolo_boxes,
    yolo_scores,
    yolo_labels,
    yolo_id2name,
    keep_label="crystal",
    overlap_thresh=0.4,
):
    """Return the SAHI PredictionResult updated with fused boxes from YOLOv12 core model.
    Required input is the original PredictionResult object from SAHI, and the yolo boxes, scores and labels as numpy arrays
    and also the yolo id to name lookup.
    """

    sahi_boxes, sahi_scores, sahi_names = [], [], []
    for p in sahi_result.object_prediction_list:
        sahi_boxes.append(p.bbox.to_xyxy())
        sahi_scores.append(p.score.value)
        sahi_names.append(p.category.name)

    # Run asymmetric NMS
    fused_boxes, fused_scores, fused_src = asymmetric_nms(
        yolo_boxes, yolo_scores, sahi_boxes, sahi_scores, overlap_thresh
    )

    # Build new ObjectPrediction list
    final_preds = []
    full_shape = read_image(image_path).shape

    for box, score, src in zip(fused_boxes, fused_scores, fused_src):
        x1, y1, x2, y2 = map(float, box)

        if src == 0:  # YOLO origin
            idx = np.where(
                (yolo_boxes == box).all(axis=1)
            )[0][0]  # find original index
            label_name = yolo_id2name[yolo_labels[idx]]
            if label_name != keep_label:
                continue  # discard non-crystal YOLO labels

        else:  # SAHI origin
            idx = np.where(
                (sahi_boxes == box).all(axis=1)
            )[0][0]
            label_name = sahi_names[idx]

        final_preds.append(
            ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                score=score,
                category_id=0,            # single class case
                category_name=label_name,
                full_shape=full_shape,
            )
        )

    sahi_result.object_prediction_list = final_preds
    return sahi_result

def compute_iou(boxA, boxB):
    '''
    Compute IoU between two boxes: [x1, y1, x2, y2]
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = min(areaA, areaB)

    return inter / union if union > 0 else 0.0

def apply_standard_nms_to_sahi(sahi_result, iou_thresh=0.3):
    '''
    Takes a PredictionResult object generated from SAHI and applies a standard
    Non max suppression to it, returning another PredictionResult with the filtered
    bounding boxes
    '''
    preds = sahi_result.object_prediction_list
    if len(preds) <= 1:
        return sahi_result  # nothing to suppress

    # Sort by confidence
    preds = sorted(preds, key=lambda p: p.score.value, reverse=True)

    keep = []
    while preds:
        current = preds.pop(0)
        keep.append(current)

        preds = [
            other
            for other in preds
            if compute_iou(
                current.bbox.to_xyxy(), other.bbox.to_xyxy()
            )
            < iou_thresh
        ]

    return PredictionResult(object_prediction_list=keep, image=sahi_result.image)



'''
runs the entire detection and segmentation pipeline on a given image path
'''


class Pipeline:
    def __init__(self, detection_model_path, segmentation_model_path, imgsz=960, conf_thresh=0.52, slice_width=300, slice_height=300, slice_overlap=0.2):
        self.model_detect = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=detection_model_path,
            confidence_threshold=conf_thresh,
            device="cpu",
            image_size=imgsz,
        )
        self.solo_model = YOLO(DETECT_MODEL)
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.slice_overlap = slice_overlap
        self.model_segment = YOLO(segmentation_model_path)
        self.imgsz = imgsz
        self.i = 1
    
    def generate_detections(self, image_path, export_dir):
        '''
        Generates detection result by combining SAHI and Yolov12 results, and stores the
        original image as orig_img, the image name as img_name and the final PredictionResult sahi object as
        the attriubte self.res_detect
        The export_dir is a redundant attribute
        '''
        result = get_sliced_prediction(
            detection_model=self.model_detect,
            image= image_path,
            slice_height =self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.slice_overlap,
            overlap_width_ratio=self.slice_overlap,
            postprocess_type='NMM'
        )
        result = apply_standard_nms_to_sahi(result)
        res_yolo = self.solo_model.predict(image_path, imgsz=960, conf=0.43)[0]
        y_boxes, y_scores, y_labels, id2name = (
            res_yolo.boxes.xyxy.numpy(),        # (N,4) xyxy
            res_yolo.boxes.conf.numpy(),        # (N,)  scores
            res_yolo.boxes.cls.numpy().astype(int),  # (N,)  class indices
            self.solo_model.names,                         # dict id -> name
        )
        fused_result = fuse_into_sahi(image_path, result, y_boxes, y_scores, y_labels, id2name, keep_label="crystal", overlap_thresh=0.4)
        img = cv2.imread(image_path)
        self.orig_img = img
        self.res_detect = fused_result
        self.img_name = image_path.split('/')[-1]
        fused_result.export_visuals(export_dir=export_dir)

    
    def display_detections(self):
        '''
        Returns the original image overlayed with bounding boxes corresponding to the detected objects.
        Also returns a list of detected crystals, each a numpy array that represents a cropped section of the
        original images
        This list is also stored in self.det_list
        '''
        lst = self.res_detect.object_prediction_list
        cv2_image = np.array(self.res_detect.image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        out_list = []
        for i in range(len(self.res_detect.object_prediction_list)):
            if lst[i].category.id == 0:
                out_list.append(cv2_image[max(0, int(lst[i].bbox.miny) - 1): int(lst[i].bbox.maxy) +1, max(0, int(lst[i].bbox.minx) -1):int(lst[i].bbox.maxx)+1])
        
        cv2_image_2 = np.array(self.res_detect.image)
        cv2_image_2 = cv2.cvtColor(cv2_image_2, cv2.COLOR_RGB2BGR)
        for i in range(len(self.res_detect.object_prediction_list)):
            if lst[i].category.id == 0:
                cv2.rectangle(cv2_image_2, (int(lst[i].bbox.minx), int(lst[i].bbox.miny)), (int(lst[i].bbox.maxx), int(lst[i].bbox.maxy)), (0, 255, 0), 2)
        self.det_list = out_list
        return cv2.resize(cv2_image_2, (self.orig_img.shape[1], self.orig_img.shape[0])), out_list
    
    
    def segment(self, out_dir):
        '''
        stores segmented mask outputs of each crystal in the out_dir,
        and returns:-
        1. a dictionary lookup with keys as the filename of a crystal mask and value as its size in sq. pixels
        2. a list of sizes in sq. pixels
        '''
        res = self.model_segment.predict(self.det_list)
        sizes = []
        size_lookup = {}
        for obj in res:
            if(obj.masks):
                obj_mask = np.array(obj.masks.data[int(np.argmax(obj.boxes.conf))])
                plt.imsave(f'{out_dir}/obj_{self.img_name}_{self.i}.jpg',cv2.resize(obj_mask, (obj.orig_shape[1], obj.orig_shape[0]))*cv2.cvtColor(obj.orig_img, cv2.COLOR_BGR2GRAY))
                self.i += 1
                size = (float)(obj.orig_shape[1] * obj.orig_shape[0] * obj_mask.sum())/(obj_mask.shape[0]*obj_mask.shape[1])
                size = size * float(self.orig_img.shape[0] * self.orig_img.shape[1])/float(self.imgsz*self.imgsz)
                sizes.append(size)
                size_lookup[f'{out_dir}/obj_{self.img_name}_{self.i}.jpg'] = float(size)
        return size_lookup, sizes
    
    


    def export_detections_to_csv(self, csv_path="resultats_detections.csv"):
        # Vérification que self.res_detect est défini
        if not hasattr(self, 'res_detect') or self.res_detect is None:
            print("Aucune détection à exporter.")
            return

        data = []

        # Générer une ligne par détection
        for pred in self.res_detect.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            data.append({
                "image_name": self.res_detect.image_path.split('/')[-1],
                "category": pred.category_name,
                "score": pred.score.value,
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3]
            })

        if not data:
            print("Aucune prédiction dans self.res_detect.")
            return

        df_new = pd.DataFrame(data)

        # Charger et concaténer si le fichier existe
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Sauvegarde finale
        df_combined.to_csv(csv_path, index=False)
        print(f"{len(data)} lignes ajoutées à {csv_path}")
            








def remove_outliers(data):
    '''
    used to filter sizes using iqr
    '''
    if len(data) < 4:
        return data  # Not enough data to determine outliers

    sorted_data = sorted(data)
    q1 = sorted_data[len(sorted_data) // 4]
    q3 = sorted_data[(len(sorted_data) * 3) // 4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return [x for x in data if lower_bound <= x <= upper_bound]