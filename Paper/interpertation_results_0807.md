# Model Evaluation on the Validation Dataset (08/07/2025)

We evaluated our Mask R-CNN model on the validation dataset using the COCO evaluation protocol, reporting both bounding box detection and instance segmentation performance.

## Results

| Metric                   | BBox (Detection) | Segm (Segmentation) |
| ------------------------ | ---------------- | ------------------- |
| **AP (IoU=0.50:0.95)**   | 66.7             | 68.9                |
| **AP50 (IoU=0.50)**      | 88.5             | 89.2                |
| **AP75 (IoU=0.75)**      | 80.4             | 82.4                |
| **APs (small objects)**  | 56.6             | 56.4                |
| **APm (medium objects)** | 64.0             | 66.6                |
| **APl (large objects)**  | 78.4             | 81.0                |

## Interpretation

The **mean Average Precision (mAP) across IoU thresholds from 0.50 to 0.95 is 68.9 for segmentation and 66.7 for detection**, demonstrating robust model performance on this dataset. High AP scores at IoU=0.50 (~89%) indicate that the model reliably detects most objects, while strong performance at IoU=0.75 (~82%) reflects good localization precision.

Performance across object sizes shows **higher accuracy on large objects (APl ~81%)**, while performance on small objects is comparatively lower (APs ~56%), a common outcome in instance segmentation due to resolution constraints and the inherent difficulty of segmenting small objects.

It is important to note that the **dataset was manually labeled, and annotations may not be perfectly accurate**, which can affect the reported mAP values. Slight inaccuracies in the ground-truth masks or bounding boxes, particularly around object boundaries, may penalize the IoU-based evaluation metrics even when the model produces visually satisfactory segmentations. Additionally, systematic ambiguities in labeling may have contributed to certain detection discrepancies observed during qualitative inspection.

## Conclusion

Overall, these results indicate that the model achieves **reliable instance segmentation performance on the dataset despite minor imperfections in manual labeling**. The high AP scores on large objects confirm the model’s utility for identifying and segmenting the primary targets of interest, while the reasonably good performance on smaller instances demonstrates its adaptability across object scales. These outcomes support the model’s readiness for further application on the test dataset and downstream quantitative analysis.
