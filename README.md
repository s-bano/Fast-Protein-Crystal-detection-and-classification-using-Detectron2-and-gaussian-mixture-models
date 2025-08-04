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

## 📝 Basic Requirements

- Python 3. - Needed for everything from dataset creation to process images.
- Preferably being on an UNIX based OS as this project hasn't been tested on Windows yet and some dataset building programes may not work correctly on it.

## 💾 Running Crystal Detection with Google Colab

We provide a ready-to-use **Google Colab notebook** to run the crystal detection and segmentation model on your image dataset without local setup.

### Features

- Process an entire ZIP archive containing images (including nested folders).
- Outputs a ZIP archive preserving the folder structure, with:
  - Images annotated with detected crystals.
  - CSV files listing crystal sizes in pixels.

### How to use

1. Open the [Colab notebook](https://colab.research.google.com/drive/1SlUv-KsLp-mUdxJQZ1cdwdg4xdY7QlKu?usp=sharing).
2. Make a **copy** of the notebook to your own Google Drive.
3. Upload your images ZIP archive and your `.pth` model file.
   > [Here]() is the link to the latest model trained by us.  
   > If you want to train your own model, please go to [Crystal Detection Training](#crystal-detection-training)
4. Edit the example paths in the notebook to match your files.
5. Run all cells (`Runtime > Run all`) to process your images.

---

### Notes

- The notebook supports recursive folders inside the ZIP archive.
- A pre-trained model by Raphaël Kuhn is provided, or you can train your own.
- The output ZIP will mirror your input folder structure.
- In the future, the aim is to deploy this app so it can be run on local devices and after that on microfluidic devices (Go [here](#deployment) for more infos)

## 📝 Crystal Detection Training

This section is only if you want to train your own Detectron2 model for crystal detection and segmentation.
To simply run the crystal detection on a set of images, please go [here](#running-crystal-detection-with-google-colab)

### Requirements

- A dataset of labeled crystal images (COCO format)

  > If you want to build your own dataset, please refer to the [Dataset Building](#dataset-building) section for instructions and resources.  
  > Otherwise, you can download the latest labeled dataset created by Raphaël Kuhn [here](insert_link).

- Detectron2 installation guide: [Download here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

  > If you struggle to install detectron2, prefer using the ready-to-use [Google Colab script](#train-on-google-colab) to do the training instead of the local script

- A CUDA-compatible GPU is recommended (this script has not been tested on CPU)

### Train on Google Colab

1. Open the [Colab notebook](insert_link).
2. Make a **copy** of the notebook to your own Google Drive.
3. Upload your dataset ZIP archive.
   > [Here]() is the link to the latest dataset built by us.
4. Edit the example paths in the notebook to match your files.
5. Run the designated cells
6. (Optionnal) Evaluate the performances of your newly trained model or export it using the dedicated section in the colab notebook

### Train on local device

Not supported yet...

## 🧠 Dataset Building

All the scripts required for this section are located in the `database_build` folder of this repository.

0. Install all required Python packages using the following command.  
   If this fails, you can manually check the dependencies listed in [requirements.txt](requirements.txt):

   ```bash
   pip install -r requirements.txt
   ```

1. **Labeling**  
   Use [Labelme](https://github.com/wkentaro/labelme) (or any compatible tool) to annotate your images.  
   Delimit each crystal with a polygon and assign it the class name `cristal`, then save the annotations in `.json` format.

2. **(Optional) Organize JSON Files**  
   You can use the script below to move all `.json` files from one folder to another:

   ```bash
   python move_json.py [source_folder] [output_folder]
   ```

3. **Build the Dataset**  
   Finally, run the script below to automatically generate a dataset in COCO format with all necessary annotation files:

   ```bash
   python build_detectron_dataset.py [jsons_folder] [images_folder] [output_folder_name]
   ```

## 🔬 Crystal Classification (WIP)

The goal of this section is to **perform unsupervised classification of detected crystals** by extracting meaningful visual features.

This involves analyzing the segmented crystals and grouping them into different classes **without prior labeling**, using techniques such as clustering or dimensionality reduction based on extracted descriptors.

## 🧪 Deployment

This section focuses on **integrating AI models with microfluidic devices to enable automated, real-time crystal measurement**.

The first step is to run the crystal detection script on local devices. Currently, running the script requires Detectron2, which can be challenging to install. Therefore, the immediate goal is to export the trained Detectron2 model to TorchScript for easier deployment.

The long-term objective is to deploy crystal detection and classification directly on microfluidic devices, which will bring additional challenges to address.

## 🚧 Roadmap

- Validate the AI-enhanced measurement system through extensive testing.
- Properly document the Google Colab scripts.
- Export models and scripts to enable local execution.
- Perform unsupervised classification of crystals.
- Deploy the full pipeline in real-world protein crystallography experiments.
