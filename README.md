# plant-disease-detection
# Plant Village Disease Detection using PyTorch & Transfer Learning

This repository contains a deep learning pipeline for identifying plant leaf diseases using computer vision and transfer learning techniques.

## 🚀 Technologies Used
* **Python** & **PyTorch**
* **Timm (PyTorch Image Models)** - ResNet50 Architecture
* **Scikit-Learn** (Evaluation metrics & Confusion Matrix)
* **Google Colab** (Training environment)

## 📊 Methodology & Pipeline
1. **Dataset:** PlantVillage dataset (`color` subset) containing multiple plant species and disease classes.
2. **Data Augmentation:** Applied `RandomResizedCrop`, `RandomHorizontalFlip`, and `ColorJitter` to ensure robust training.
3. **Two-Stage Fine-Tuning:**
   * **Stage 1:** Frozen backbone, trained only the classification head (~84% validation accuracy).
   * **Stage 2:** Unfroze all layers and applied fine-tuning with a reduced learning rate (`1e-4`) using `AdamW` and `OneCycleLR`.
4. **Evaluation:** Detailed classification report, macro/weighted F1-scores, and confusion matrix analysis to identify misclassified plant classes.

## 📂 Files in Repository
* `First_Github_Using.ipynb`: Complete Jupyter Notebook containing training, evaluation, and analysis code.
* **Pre-trained Model Weights:** Since the model file (`.pt`) is large (~92 MB), you can download the fine-tuned weights directly from here: [Google Drive Download Link](https://drive.google.com/file/d/1ExAKKgqxcJiv6cTuVvY3wVbXnReL8aCo/view?usp=sharing)
