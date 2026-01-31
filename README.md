# Titanic-Survival-Prediction-ML-and-data-science-project
Machine learning project to predict Titanic passenger survival using data preprocessing, multiple classifiers, ROC analysis, and hyperparameter tuning.

This project predicts whether a passenger survived the Titanic disaster using machine learning.
It demonstrates a complete ML pipeline including data preprocessing, model training, evaluation, ROC analysis, and hyperparameter tuning.

📌 Objective

To build and compare multiple machine learning models to accurately predict Titanic passenger survival and identify the best performing algorithm.

📊 Dataset

891 passenger records

12 original features

Target variable: Survived (0 = Did Not Survive, 1 = Survived)

🔧 Data Preprocessing

Dropped Cabin column (77% missing values)

Imputed missing values:

Age → median

Embarked → mode

Encoded categorical variables (Sex, Embarked)

Scaled numeric features for KNN & SVM

🤖 Machine Learning Models

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree

Naive Bayes

Support Vector Machine (SVM)

Linear Regression (converted to classifier)

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC AUC

🏆 Best Performing Models
Model	Validation AUC	Test AUC
Logistic Regression	0.86	0.86
KNN	0.85	0.85
Decision Tree	0.86	0.84
⚙️ Hyperparameter Tuning

Used GridSearchCV (5-fold cross validation) to tune:

KNN → n_neighbors, weights

Logistic Regression → C, solver

SVM → C, kernel, gamma

Decision Tree → max_depth, min_samples_split

Naive Bayes → var_smoothing

📊 Visualizations

Survival Distribution Bar Chart

Correlation Heatmap

ROC Curves (Validation & Test)

