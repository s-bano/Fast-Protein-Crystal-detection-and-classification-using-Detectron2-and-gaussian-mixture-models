Im just going to include a short description of how the current pipeline works.
But first, ill write down the specifications:-

# Specifications

1. YOLOv12-n detection 
weights - ./models/best8-detect.pt
fine tuned on - dataset-detect (inclued in the folder)
for - about 350 epochs 
imgsz - 960*960
confidence threshold - 0.43

2. SAHI pipeline
base model - yolov12 mentioned above
imgsz - 960
confidence threshold - 0.52
post processing - NMM
slice width - 300
slice height - 300
slice overlap 0.20

3. Results merging
sahi internal standard nms overlap threshold - 0.3
asymmetric nms ios overlap threshold - 0.4

4. Segmentation model
model - yolo11n-seg (./models/seg-final-yolo.pt)
imgsz - 320


# PIPELINE EXPLANATION - 

- we take the image and make two detection results from it:-

1. We send the image through a fine tuned YOLOv12 model
2. We send the same image through a slicing aided hyper inference (SAHI) pipeline that uses the same YOLOv12 model

- This enables us to detect a large number of smaller crystals, even in a noisy environment and large scale crystals

Now, how the YOLOv12 model has been trained is as follows:-
- I used a sort of contrastive labelling. So, it not only detects crystals but it also tries to detect areas of crystal agglomerates in the image
- The individual crystals are labelled 'crystal' and the overlapping/agglomerated areas are labelled as non-crystal.
Training on this task makes it choose only individual crystals.

However, when we run the sahi pipeline, due to detecting small scale features it might detect a small part of a crystal as a seperate crystal and might also detect the section of an agglomerate as a crystal.
To avoid this, we augment the result of sahi using the YOLOv12 base model detections.

Before doing this though, we apply standard NMS to the sahi results to remove all overlaps.

We take the yolov12 detections and the sahi prediction result and apply asymmetric non max suppresion, which means,
all sahi detected bounding box having a significant overlap with any large object detected from the base yolov12 model (crystal as well as non crystal) is suppressed, or removed. the criteria we use for such a suppression is not the standard iou, but ios, which is intersection over area of the smaller object, which works much better in this case.

now we fuse the remaining sahi results with the yolo results. Also, we trained a yolo segmentation model on a custom dataset of detected crystals.
after detection is done, a list of cropped detected crystals ie extracted and this list is sent as a batch through the segmentation model. after we get the segmentation mask, we apply some simple conversions to work our way back and get the size of each crystal in square pixels. this list of sizes is then stored as an excel sheet during exporting.

The sizes are filtered by +-1.5 iqr and then a histogram is prepared for the sizes.
