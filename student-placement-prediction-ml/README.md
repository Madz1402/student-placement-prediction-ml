A Reproducible Machine Learning Framework for Student Placement Prediction

An Empirical Study Using Enhanced Dataset and Preprocessing Techniques

This repository contains the complete reproducible implementation of the research work:

“A Reproducible Machine Learning Framework for Student Placement Prediction”

The goal of this project is to predict whether a student will get placed using academic performance and skill-related attributes and to compare multiple machine learning algorithms with a proposed hybrid ensemble model.

Abstract

This project implements a machine learning–based framework to predict campus placement outcomes using academic and skill-oriented features such as CGPA, projects, internships, workshops, communication skills, hackathon participation, and backlogs.
Multiple classification models including Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and XGBoost are trained and evaluated.

To improve prediction performance and reliability, a hybrid ensemble model combining Random Forest and XGBoost using soft voting is proposed.
The hybrid model achieved 93.75% accuracy, outperforming individual models and providing more stable predictions.

Project Structure
student-placement-prediction-ml/
├── dataset/
│   └── Placement_Prediction_data.csv
├── notebooks/
│   └── placement_prediction.ipynb
├── src/
│   └── train_model.py
└── README.md

Dataset Description

The dataset contains the following features:

Feature	Description
CGPA	Academic performance
Major Projects	Number of major projects
Mini Projects	Number of mini projects
Workshops/Certifications	Technical exposure
Skills	Skill level
Communication Skill Rating	Communication ability
Internship	Internship experience
Hackathon	Hackathon participation
12th Percentage	Higher secondary score
10th Percentage	Secondary score
Backlogs	Academic backlogs
PlacementStatus	Target variable (Placed / Not Placed)
Methodology

Data preprocessing

Label encoding of categorical attributes

Feature scaling using standardization

Train-test split (80:20)

Model implementation

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

XGBoost

Hybrid Ensemble Model

Random Forest + XGBoost using soft voting

Evaluation

Accuracy

Confusion Matrix

Precision, Recall, F1-score

Results
Model	Accuracy
Logistic Regression	88.37%
Decision Tree	97.67%
Random Forest	90.70%
SVM	83.72%
KNN	79.07%
XGBoost	92.55%
Hybrid (RF + XGBoost)	93.75%

The hybrid ensemble model achieved the best balance between accuracy, stability, and generalization.

How to Run
Using Notebook

Open notebooks/placement_prediction.ipynb in Google Colab

Upload dataset/Placement_Prediction_data.csv

Run all cells

Using Python Script
pip install -r requirements.txt
python src/train_model.py

Author

D.S. Madhumitha
M.Sc. Software Systems (CT-PG)
Kongu Engineering College