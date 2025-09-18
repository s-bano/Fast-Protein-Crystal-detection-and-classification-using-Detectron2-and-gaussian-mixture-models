# AI-CrystalMetrics

AI-Driven Measurement of Protein Crystals on Microfluidic Chips

This project is set of tools use to detect with segmentation cristals on images and then classify them into auto-build classes.

<!-- The primary aim of this project is to leverage artificial intelligence (AI) to enhance the precision and efficiency of measuring diffractable protein crystals on a microfluidic chip. This involves developing and implementing AI algorithms (CNN) that can accurately identify and analyze protein crystals, thereby improving the overall process of protein crystallography.
The specific objectives of the project are:

- **Algorithm Development**: To design and train AI models that can accurately detect and measure protein crystals on a microfluidic chip.
- **System Integration**: To integrate the AI models with microfluidic devices, enabling automated and real-time crystal measurement.
- **Data Analysis**: To analyze the performance of the AI models in terms of accuracy, speed, and reliability, and compare these metrics with traditional measurement techniques.
- **Optimization**: To optimize the AI algorithms for better performance and adaptability to different types of protein crystals.
- **Validation**: To validate the AI-enhanced measurement system through extensive testing and application in actual protein crystallography experiments. -->

---

## 💾 Running Crystal Detection with Google Colab

We provide a ready-to-use **Google Colab notebook** to run the crystal detection and segmentation model on your image dataset without local setup.

> You can check the video tutorial [here](https://youtu.be/Q-XzCx8v-h8)

### Features

- Process an entire ZIP archive containing images (including nested folders) or single images.
- Features a graphical web environment
- Outputs a ZIP archive preserving the folder structure, with:
  - Images annotated with detected crystals.
  - XLSX files listing crystal sizes (in pixels and um) and predicted classes.

### How to use

1. Open the [Colab notebook](https://colab.research.google.com/drive/1SlUv-KsLp-mUdxJQZ1cdwdg4xdY7QlKu?usp=sharing)
2. Run all cells (`Runtime > Run all`) to launche the graphica environment (It may take a while since it's also downloading resources).
3. Upload your images or ZIP files and click `Run`

---

## 📝 Crystal Detection Training

[here](https://drive.google.com/file/d/1hAQAWmmDCPmKggARMJYlSXfMqIOfNMTI/view?usp=sharing) is the link to the orginal dataset of non-labelized images.

### Requirements

- Python >= 3.12 - Needed for everything from dataset creation to process images.
- Preferably being on an UNIX based OS as this project hasn't been tested on Windows yet and some dataset building programes may not work correctly on it.

This section is only if you want to train your own Detectron2 model for crystal detection and segmentation.
To simply run the crystal detection on a set of images, please go [here](#running-crystal-detection-with-google-colab)

- A dataset of labeled crystal images (COCO format)

  > If you want to build your own dataset, please refer to the [Dataset Building](#dataset-building) section for instructions and resources.  
  > Otherwise, you can download the latest labeled dataset created by Raphaël Kuhn [here](https://liverguac-my.sharepoint.com/:f:/g/personal/r_kuhn_rgu_ac_uk/Eqou7erMVKBNpnR7u5lhXRQBeREYnuNjxXD12QRsb1UFqg?e=Pv7mB7).

- Detectron2 installation guide: [Download here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

  > If you struggle to install detectron2, prefer using the ready-to-use [Google Colab script](#train-on-google-colab) to do the training instead of the local script

- A CUDA-compatible GPU is recommended (this script has not been tested on CPU)

### Train on Google Colab

1. Open the [Colab notebook](https://colab.research.google.com/drive/1BVv4seAGRw9qVKoPWAsBJ_uWEBzY9U14?usp=sharing).
2. Make a **copy** of the notebook to your own Google Drive.
3. Upload your dataset ZIP archive.
   > [Here]() is the link to the latest dataset built by us.
4. Edit the example paths in the notebook to match your files.
5. Run the designated cells
6. (Optionnal) Evaluate the performances of your newly trained model or export it using the dedicated section in the colab notebook

---

## 🧠 Dataset Building

All the scripts required for this section are located in the `database_build` folder of this repository.

0. Install all required Python packages using the following command.  
   If this fails, you can manually check the dependencies listed in [requirements.txt](requirements.txt):

   ```bash
   pip install -r requirements.txt
   ```

1. **(Optional) Rearrange images names**
   Use this script to correctly name all your images

   ```bash
   python flatten_names.py [source_folder] [output_folder]

   ```

2. **Labeling**  
   Use [Labelme](https://github.com/wkentaro/labelme) (or any compatible tool) to annotate your images.  
   Delimit each crystal with a polygon and assign it the class name `cristal`, then save the annotations in `.json` format.

3. **(Optional) Organize JSON Files**  
   You can use the script below to move all `.json` files from one folder to another:

   ```bash
   python move_json.py [source_folder] [output_folder]
   ```

4. **Build the Dataset**  
   Finally, run the script below to automatically generate a dataset in COCO format with all necessary annotation files:

   ```bash
   python build_detectron_dataset.py [jsons_folder] [images_folder] [output_folder_name]
   ```

---

## 🔬 Crystal Classification with Guassian Model Mixture

The goal of this section is to **perform unsupervised classification of detected crystals** by extracting meaningful visual features.

This involves analyzing the segmented crystals and grouping them into different classes **without prior labeling**, using techniques such as clustering or dimensionality reduction based on extracted descriptors. A GMM model is already used by the Crystal Detection Colab Notebook.

### Train a GMM model

1. Extract features using this google Colab notebook into a `h5` format
2. To get a new `gmm_model`, use
   ```bash
   python gmm_training.py features.h5
   ```

## AICM Python Package

The Google Colab Notebook use a custom python package called AICM containing the detectron2 and GMM models and scripts containing usefull functions.
It is automaticaly downloaded in the Colab Notebook

### Build

If you want to build a new AICM package, you can specify the path to the files you want to include in `build_package.sh` and then launch it with:

```bash
chmod u+x build_package.sh
./build_package.sh
```
