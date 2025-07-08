# 🧬 Skin Lesion Classification with a Novel CNN Pipeline

This project presents a novel multi-model Convolutional Neural Network (CNN) architecture for early skin cancer detection. Inspired by the importance of at-home diagnostic tools, we designed a pipeline that classifies skin abnormalities into both cancerous and non-cancerous categories — and further classifies them by type.

The research report can be found: [A Multi-Model Approach to Skin Lesion Analysis with Machine Learning](https://github.com/Aarya-Kul/Skin_Cancer_Detection/blob/main/Research_Report.pdf) 

---

## 🏗️ Project Overview

We created a **three-model pipeline** to improve interpretability and diagnostic accuracy:

1. **Model 1 – Cancer Detector**
   - Binary classifier: benign vs malignant
   - Uses dropout, max pooling, and tuned learning rate
   - Achieved **89.07% test accuracy**

2. **Model 2 – Cancer Type Classifier**
   - Classifies malignant lesions as **melanoma** or **basal cell carcinoma**
   - Employs residual blocks for deeper learning
   - Achieved **87.27% test accuracy**

3. **Model 3 – Benign Type Classifier**
   - Multi-class classifier for 5 types of benign lesions
   - Uses batch norm, dropout, and softmax activation
   - Achieved **84.13% test accuracy**

Each model is implemented using **PyTorch**, and optimized for limited Colab GPU resources.

---

## 🧠 LLM Integration (Bonus)

After classification, we use **AdaptLLM/medicine-chat** and **Google Gemini** to explain the predicted condition to the user using a generated natural language response.

**Steps:**
- Parse age, gender, and location metadata
- Construct a patient-specific prompt
- Query the LLMs for tailored medical advice

---

## 🧪 Dataset and Preprocessing

- Dataset: **HAM10000** (Harvard)
- Preprocessing: rotation, Gaussian noise, normalization
- Custom PyTorch `Dataset` and `DataLoader` handle:
  - Image loading & label parsing
  - Metadata filtering (cancerous / benign splits)
- Downsampling techniques used to reduce class imbalance

---

## 📊 Evaluation Summary

| Model   | Task                         | Test Accuracy |
|---------|------------------------------|---------------|
| Model 1 | Cancer vs Non-Cancer         | 89.07%        |
| Model 2 | Melanoma vs BCC              | 87.27%        |
| Model 3 | Multi-Class Benign Lesions   | 84.13%        |

---

## 📂 Repository Contents

```
├── SkinLesionML.ipynb           # Full notebook with model training and inference
├── Research_Report.pdf          # Full report with methodology, results, and figures
├── data/                        # (Expected) folder for HAM10000 images and CSVs
├── models/                      # (Optional) for saving model weights
```

---

## 🚀 Running the Notebook

1. Upload the HAM10000 dataset into the `data/` directory.
2. Launch `SkinLesionML.ipynb` on Google Colab (recommended).
3. Run all cells — results and classification responses will be displayed.

---

## ✅ Key Contributions

- Modular pipeline for better lesion interpretability
- Custom CNNs for limited-resource environments
- Residual blocks boost deep lesion classification
- Integrated LLM for user-facing disease explanation

---

## 🧭 Future Work

- Integrate Vision Transformers (ViT) for advanced pattern learning
- Improve dataset balance (add darker skin tones)
- Fine-tune LLM prompt logic for medical accuracy
- Explore temporal lesion changes via sequential image inputs

---

**Team Members:** Aarya Kulshrestha, Bowen Yi, Christopher Erndteman, Maaz Hussain, Shankhin Chirmade
