# Processus de création de la base de données

L'objectif de cette barnche du projet est de créer des datasets prets a etres utilises dan sun contexte de reseau de neurones.
Plus precisement 2 types de datasets:

- Dataset pour des U-Net
- Dataset pour des R-CNN

Pour chaque dataset, on part des donnés labelises sous foreme de `.json` via le logiciel `labelme`
1013 images dans la base de données

## 0. Usage

Lancer `labelme` pour dessiner chaque cristal de chaque image du dossier `images`

```bash
labelme
```

### 1. Procédure automatisé

Make sure you're positionned in `database_build` folder then run:

```bash
./build_database.sh -u     # -u pour une base de donneés adpatée au U-Net
./build_database.sh -r     # -r pour une base de donnés adaptée au R-CNN
./build_database.sh -a     # Pour générer les 2
```

### 2. Procédure manuelle

Générer les 2 dossiers differents de masques pour les 2 usages (U-Net et R-CNN) qui n'ont pas les memes besoins

```bash
python json2masks.py jsons images multi_masks
python masks2single.py multi_masks single_masks
```

Générer le fichier `annotations.json` nécéssaire pour les reseau de neruones de type R-CNN

Convertir les dossiers générés en archives incluants toutes les donnees pour chaque type de dataset

```bash
python build_u-net_dataset.py images_folder single_masks_folder
python build r-cnn_dataset.py images_folder multi_masks_folder
```

## 📦 1. Base de données pour U-Net (Segmentation sémantique)

Pour les U-Net a chaque image est associe un masque de labelisation, qui dit pour chaque pixel si il fait partie d un cristal ou non.

📁 Structure finale :

```bash
u-net_dataset/
├── images/
│   ├── image_001.png
│   └── ...
├── masks/
│   ├── image_001_mask.png
│   └── ...
```

✅ Utilisable directement avec U-Net (Keras, PyTorch, etc.).

📁 Procédure :

```bash
python json2masks.py jsons images multi_masks
python masks2single.py multi_masks single_masks
```

⸻

## 🧠 2. Base de données pour Mask R-CNN (Segmentation d’instance)

✅ À faire :

1. Partir du même JSON LabelMe.
2. Utiliser le script labelme_to_instance_masks.py pour créer un masque par polygone :
   • mask_001.png, mask_002.png, etc.
   • Chaque image a son propre sous-dossier de masques

📁 Structure attendue :

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
├── annotations.json  ← à générer (prochaine étape)
```

⸻

## 🔧 Étape suivante : Générer le annotations.json (COCO)

2 méthodes:

- open-source coco-maskgen
- script perso basé sur pycocotools.

# Database Creation Process

The goal of this project branch is to create datasets ready to be used in a neural network context.
Specifically, two types of datasets:

- Dataset for U-Net
- Dataset for R-CNN

For each dataset, we start from labeled data in `.json` format created using the `labelme` software.
1013 images in the database

## 0. Usage

Launch `labelme` to draw each crystal in each image in the `images` folder

```bash
labelme
```

### 1. Automated Procedure

Make sure you're positioned in the `database_build` folder then run:

```bash
./build_database.sh -u     # -u for a dataset adapted to U-Net
./build_database.sh -r     # -r for a dataset adapted to R-CNN
./build_database.sh -a     # To generate both
```

### 2. Manual Procedure

Generate the two different mask folders for the two usages (U-Net and R-CNN), which have different needs

```bash
python json2masks.py jsons images multi_masks
python masks2single.py multi_masks single_masks
```

Generate the `annotations.json` file needed for R-CNN type neural networks

Convert the generated folders into archives including all data for each dataset type

```bash
python build_u-net_dataset.py images_folder single_masks_folder
python build r-cnn_dataset.py images_folder multi_masks_folder
```

## 📦 1. U-Net Dataset (Semantic Segmentation)

For U-Net, each image is associated with a labeling mask, indicating for each pixel whether it is part of a crystal or not.

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
2. Use the script labelme_to_instance_masks.py to create one mask per polygon:
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
├── annotations.json
```

⸻

## 🔧 Next Step: Generate annotations.json (COCO)

2 methods:

- open-source coco-maskgen
- custom script based on pycocotools
