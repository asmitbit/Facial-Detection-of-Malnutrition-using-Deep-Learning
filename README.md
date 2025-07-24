# Malnutrition Detection Using Facial Analysis  

*A Summer Internship Project by*  
**Asmit Adesh** - BTECH/10367/22,  
**Sumit Kumar** - BTECH/10706/22,  
**Abhigyan Sharma** - BTECH/10148/22  

FROM **BIT Mesra, Ranchi**

---
![Screenshot 1](Screenshot%202025-07-22%20201711.png)


## **Project Overview**  
This project focuses on building a **Deep Learning-based model** to detect malnutrition in children using **facial images**. The aim was to create a **binary classification system** (Healthy vs. Malnourished) that could assist in early identification of malnourishment, especially in rural and underprivileged areas.

---

## **Key Objectives**
- Collect and curate a **high-quality dataset** of healthy and malnourished children.
- Apply **data augmentation** techniques to address the limited dataset size.
- Train a **state-of-the-art CNN model (MobileNetV2)** using transfer learning.
- Fine-tune the model to achieve **maximum accuracy**.
- Evaluate the model using **accuracy, loss curves, and confusion matrices**.
- Prepare the system for a **frontend launch (Streamlit )**.

---

## **Workflow of the Project**

### **1. Data Collection**
- **Web Scraping:**  
  We scraped over **800+ images** of children from Bing and Google using Python tools like `bing_image_downloader` and `icrawler`.
- **Faculty-Provided Images:**  
  We added ~100 extra images given by our faculty mentor.
- **Manual Cleaning:**  
  Images were manually verified and cleaned for quality.
  ![Screenshot 2](Screenshot%202025-07-22%20202630.png)
  
**Dataset Structure:**
dataset/
├── healthy child face/
└── malnourished child face/


---

### **2. Dataset Splitting**
We split the dataset into **train, validation, and test sets**:
- **Training:** 615 images (70%)
- **Validation:** 153 images (15%)
- **Testing:** 194 images (15%)
This ensures balanced training and unbiased testing.
![screenshot4](./Screenshot%202025-07-24%20132456.png)
---

### **3. Data Augmentation**
To address limited data, we applied **mild data augmentation** using `ImageDataGenerator`:
- Horizontal flips  
- Small rotations (±10°)  
- Brightness and contrast adjustments  
- Zoom and shift transformations  

We initially experimented with **Albumentations**, but switched to `ImageDataGenerator` for better integration with MobileNetV2.

---

### **4. Model Architecture**
- **Initial Approach:** ResNet50 was tested but it overfit due to dataset size.
- **Final Model:** We chose **MobileNetV2** (pre-trained on ImageNet) as the backbone.

**Custom Classification Head:**
- Global Average Pooling Layer
- Dense Layer (128 units, ReLU activation)
- Dropout Layers for regularization
- Final Dense Layer (Sigmoid activation for binary classification)

---

### **5. Training**
- **Initial Training:**  
  - MobileNetV2 base layers were frozen.
  - The custom classification head was trained for **30 epochs**.
  - **Optimizer:** Adam (learning rate = 1e-4).

**Results (Initial Training):**
- Validation Accuracy: **85.62%**  
- Test Accuracy: **90.21%**

- **Fine-Tuning:**  
  - We unfroze the last 10 layers of MobileNetV2.
  - Reduced learning rate to **1e-6**.
  - Used early stopping and checkpoints.
![screenshot3](./Screenshot%202025-07-24%20132624.png)
---

### **6. Evaluation**
- Plotted **Training vs. Validation Accuracy and Loss** curves.
- Generated a **Confusion Matrix** to measure class-level performance.
- Achieved **~90% test accuracy** using the initial trained MobileNetV2 model.

---

### **7. Model Saving**
- **mobilenetv2_initial_model.keras** — Best performing model after transfer learning.
- **mobilenetv2_finetuned_model.keras** — Fine-tuned version for additional experimentation.

---

## **Tech Stack**
- **Languages:** Python
- **Libraries:** TensorFlow/Keras, NumPy, OpenCV, Matplotlib, Scikit-learn, Albumentations
- **Model:** MobileNetV2 (Transfer Learning)
- **Data Tools:** ImageDataGenerator, Custom Augmentation Pipelines

---

## **Results**
- **Best Test Accuracy:** **90.21%**
- **Validation Accuracy:** ~85%
- **Loss (Test):** 0.2148

---

## **Future Work**
- Gather a **larger, balanced dataset** to further improve performance.
- Implement **Grad-CAM (Explainable AI)** to visualize key facial regions contributing to predictions.

---

## **Contributors**
- **Asmit Adesh** — ML Model Development, Data Collection & Training,Model Training&Testing
- **Abhigyan Sharma** —  UI Design,Training and testing
- **Sumit Kumar** — Documentation, Dataset Management & Testing

---
We developed a **Streamlit-based web frontend** for this project to make it user-friendly:
- **Upload Interface:** Users will be able to upload a child’s image.
- **Real-Time Prediction:** The model will classify the image as "Healthy" or "Malnourished."
- **Visualization:** Display the prediction confidence score and possibly use **Grad-CAM** to highlight important facial regions.
- **Deployment:** We will host the Streamlit app (using Streamlit Cloud or Vercel).

**Acknowledgments**  
We would like to express our heartfelt gratitude to **Prof. Akriti Nigam** (BIT Mesra) for her invaluable guidance, continuous support, and kindness throughout the course of this project. Her encouragement and insightful suggestions were instrumental in shaping our work.
