# ML-04-K-Nearest Neighbors (KNN)

Build a simple KNN pipeline using Python, including data loading, preprocessing, feature scaling, model training, evaluation, and prediction.

## Data

Kaggle Iris Dataset: Kaggle [https://www.kaggle.com/datasets/uciml/iris](https://www.kaggle.com/datasets/uciml/iris)

## Structure

```text
project-KNN/
|
├── data-animal/
|   └── Iris.csv
|
├── classification/
|   ├── main.py
|   ├── data_loader.py
|   ├── knn_tf.py
|   ├── evaluate.py
|   └── outputs/
|       ├── 01_k_curve.png
|       ├── 02_confusion_matrix.png
|       └── predictions.csv
|
├── clustering/
|   ├── main.py
|   ├── data_loader.py
|   ├── kmeans_tf.py
|   ├── knn_tools.py
|   ├── visualize.py
|   └── outputs/
|       ├── 01_elbow.png
|       ├── 02_clusters.png
|       ├── cluster_summary.csv
|       └── clustered_animals.csv
|
├── requirements.txt
└── README.md

## Summary
This project demonstrates KNN for classification and K-Means for clustering using the Iris dataset. It includes data loading, preprocessing, model training, evaluation, visualization, and prediction through a modular Python pipeline.A