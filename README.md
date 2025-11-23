# Support-Vector-Machine-SVM-and-Model-Ensembling-with-breast-cancer-dataset
A breast cancer classifier using SVM (with GridSearchCV), compared it with four baseline models, and applied Bagging, Boosting, and Stacking ensembles to evaluate performance differences.

This project implements a complete machine learning workflow for classifying malignant vs. benign breast tumors using the **Breast Cancer dataset (`data-breast-cancer.csv`)**.  
The assignment focuses on **Support Vector Machine (SVM)** training, comparison with other classifiers, and application of **Bagging, Boosting, and Stacking** ensemble methods.

## 1. Objectives

- Load and clean the breast cancer dataset  
- Perform exploratory data analysis (EDA) and visualization  
- Train and tune an **SVM classifier** using `GridSearchCV` (kernel, C, gamma)  
- Train 4 additional models:  
  - Logistic Regression  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
- Apply 3 ensemble methods:  
  - **Bagging**  
  - **Boosting**  
  - **Stacking**  
- Compare performance across all models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## 2. Dataset

- File: 'data-breast-cancer.csv'
- Contains features describing breast cell nuclei  
- Label: **malignant vs. benign**  
- Preprocessing steps:
  - Handle missing values
  - Standardize numerical features
  - Identify & remove outliers (IQR or Z-score)

## 3. Methods

### 3.1 Exploratory Data Analysis
- Visualized distributions, correlations, and feature importance  
- Checked class balance  
- Removed extreme outliers if necessary  

### 3.2 SVM Model
- Performed hyperparameter tuning with `GridSearchCV`  
  - kernel ∈ {linear, rbf}  
  - C ∈ range of penalty values  
  - gamma ∈ range of kernel coefficients  
- Evaluated the best SVM model on the test set

### 3.3 Other Classifiers
Implemented and evaluated:
- Logistic Regression  
- Gaussian Naive Bayes  
- Decision Tree  
- Random Forest  

### 3.4 Ensemble Methods
- **Bagging**: applied with a Decision Tree base estimator  
- **Boosting**: AdaBoost / GradientBoosting  
- **Stacking**: combined multiple base models with a meta-learner  
- Compared ensemble performance vs individual models

## 4. Evaluation Metrics

For each model, reported:
- Accuracy  
- Precision  
- Recall  
- F1-score 

## 5. Key Findings 
- The tuned SVM was the strongest individual classifier, achieving high and balanced accuracy, precision, recall, and F1-score.
- Stacking achieved the overall best performance (accuracy ≈ 0.956), matching SVM while providing more stable and generalized predictions across all metrics.
- Gradient Boosting performed slightly lower (≈ 0.947) but still captured complex nonlinear patterns effectively.
- Bagging improved Decision Tree performance (≈ 0.939) by reducing variance and mitigating overfitting.
- AdaBoost provided moderate gains (≈ 0.930), performing better than simpler models but below the other ensemble methods.

## 6. Tech Stack
- Python  
- scikit-learn  
- pandas / NumPy  
- Matplotlib / Seaborn  



