# 📊 LAB 04: K-Nearest Neighbors (KNN) & K-Means Clustering

**รหัสวิชา:** 04-624-201 Machine Learning  
**ผู้จัดทำ:** ธัญพิสิษฐ์ เทพธัญญะ  

---

## Overview

This project demonstrates the implementation of two fundamental Machine Learning algorithms: **K-Nearest Neighbors (KNN)** for supervised classification and **K-Means** for unsupervised clustering. The workflow covers data preprocessing, model training, and performance evaluation through numerical metrics and terminal outputs.

---

## Folder Structure

```text
LAB04/
├── data-animal/
│   ├── database.sqlite      # ฐานข้อมูลตัวอย่าง
│   └── Iris.csv             # ชุดข้อมูลหลักที่ใช้สำหรับเทรนโมเดล
├── project-KNN/
│   ├── classification/      # โฟลเดอร์จัดเก็บโค้ดสำหรับโมเดล KNN
│   └── clustering/          # โฟลเดอร์จัดเก็บโค้ดสำหรับโมเดล K-Means
└── README.md                # เอกสารอธิบายโปรเจกต์

Dataset
The project utilizes a dataset to train and evaluate both models. The objective for KNN is to correctly classify data into predefined labels, while K-Means aims to discover hidden patterns and group similar data points together without prior labels.

Source: Iris Dataset (data-animal/Iris.csv)

Tasks
Data Exploration & Preprocessing

Feature and Target separation

Data Splitting (Training and Testing sets for KNN)

Feature Scaling (Standardizing data to ensure equal weight for distance-based algorithms)

Model Construction 1: K-Nearest Neighbors (Classification)

Initializing the KNN classifier with an optimal 'K' value.

Distance metric calculation (e.g., Euclidean distance).

Model evaluation using Accuracy Score and Confusion Matrix outputs.

Model Construction 2: K-Means (Clustering)

Determining the optimal number of clusters using the Elbow Method concepts.

Centroid initialization and iterative distance minimization.

Assigning data points to the nearest cluster centroids.

Technologies Used
Python

Pandas (Data manipulation)

Scikit-Learn (Machine Learning algorithms & scaling)

Project Replication & Commands

# 1. Check your Python installation environment
python --version

# 2. Install required data science packages
pip install pandas scikit-learn

# 3. Navigate to the project directory (Example for Clustering)
cd LAB04/project-KNN/clustering

# 4. Execute the main pipeline script
python main.py