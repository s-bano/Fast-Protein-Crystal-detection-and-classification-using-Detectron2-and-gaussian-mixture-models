# Database Creation Process

The goal of this branch of the project is to create datasets ready to be used in a neural network context.
More specifically, two types of datasets:

- Dataset for U-Net
- Dataset for R-CNN

For each dataset, we start from labeled data in `.json` format using the `labelme` software.
There are 1013 images in the database.

## 0. Usage

Launch `labelme` to draw each crystal in every image in the `images` folder:

```bash
labelme
```

### 1. Automated Procedure

Make sure you're positioned in the `database_build` folder, then run:

```bash
./build_database.sh -u     # -u for a dataset adapted to U-Net
./build_database.sh -r     # -r for a dataset adapted to R-CNN
./build_database.sh -a     # to generate both
```

### 2. Manual Procedure

Generate the two different mask folders for the two purposes (U-Net and R-CNN), which have different requirements:

```bash
python json2masks.py jsons images multi_masks
python masks2single.py multi_masks single_masks
```

Generate the `annotations.json` file required for R-CNN type neural networks.

Convert the generated folders into archives including all data for each dataset type:

```bash
python build_u-net_dataset.py images_folder single_masks_folder
python build r-cnn_dataset.py images_folder multi_masks_folder
```

## 📦 1. U-Net Dataset (Semantic Segmentation)

For U-Net, each image is associated with a labeling mask that indicates, for each pixel, whether it belongs to a crystal or not.

📁 Final structure:

```bash
u-net_dataset/
├── images/
│   ├── image_001.png
│   └── ...
├── masks/
│   ├── image_001_mask.png
│   └── ...
```

✅ Directly usable with U-Net (Keras, PyTorch, etc.).

📁 Procedure:

```bash
python json2masks.py jsons images multi_masks
python masks2single.py multi_masks single_masks
```

⸻

## 🧠 2. Mask R-CNN Dataset (Instance Segmentation)

✅ To do:

1. Start from the same LabelMe JSON.
2. Use the script `labelme_to_instance_masks.py` to create one mask per polygon:
   • mask_001.png, mask_002.png, etc.
   • Each image has its own subfolder of masks

📁 Expected structure:

```bash
mask-rcnn_dataset/
├── images/
│   ├── image_001.png
│   └── ...
├── masks/
│   └── image_001/
│       ├── mask_001.png
│       ├── mask_002.png
│       └── ...
├── annotations.json  ← to be generated (next step)
```

⸻

## 🔧 Next Step: Generate `annotations.json` (COCO format)

Two methods:

- open-source coco-maskgen
- custom script based on pycocotools
