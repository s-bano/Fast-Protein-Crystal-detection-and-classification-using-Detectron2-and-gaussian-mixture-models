# Processus de création de la base de données

1013 images dans la base de données

## 📦 1. Base de données pour U-Net (Segmentation sémantique)

✅ À faire :

1. Annoter l’image avec LabelMe en traçant tous les cristaux dans le même JSON (un shape par cristal).
2. Fusionner tous les polygones pour créer 1 seul masque binaire par image :
   • Fond = 0
   • Cristaux = 255 (ou 1)

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
