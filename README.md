# Titanic-Survival-Prediction-ML-and-data-science-project
This project predicts whether a passenger survived the Titanic disaster using multiple machine learning algorithms. It demonstrates the full data science workflow: data exploration, preprocessing, feature engineering, model training, evaluation using ROC AUC, and hyperparameter tuning.

📌 Problem Statement

The Titanic dataset is a classic binary classification problem. Given information about a passenger (age, gender, ticket class, fare, etc.), the goal is to predict whether the passenger survived (1) or not (0).

📊 Dataset

891 passenger records

12 original features

Target: Survived (0 = Did not survive, 1 = Survived)

🔧 Data Preprocessing

Dropped Cabin column (77% missing values)

Filled missing values:

Age → median

Embarked → mode

Converted categorical variables to numeric using encoding

Scaled numerical features for distance-based models

🤖 Models Explained
1️⃣ Logistic Regression

Type: Linear classifier
How it works:
Logistic Regression models the probability that a passenger survives using a sigmoid function. It finds a linear boundary that separates survivors and non-survivors.

Why used:

Simple

Fast

Highly interpretable

Works well for binary classification

2️⃣ K-Nearest Neighbors (KNN)

Type: Distance-based classifier
How it works:
KNN predicts a passenger’s survival based on the majority class of the k closest passengers in the dataset.

Why used:

No training phase

Captures complex patterns

Sensitive to feature scaling (so scaling is required)

3️⃣ Decision Tree

Type: Rule-based classifier
How it works:
The model creates a tree of decision rules based on feature values (e.g., gender, class, age) to predict survival.

Why used:

Easy to interpret

Handles non-linear data

Shows feature importance

4️⃣ Naive Bayes

Type: Probabilistic classifier
How it works:
Uses Bayes’ Theorem and assumes independence between features to compute the probability of survival.

Why used:

Very fast

Works well with small datasets

Handles noise well

5️⃣ Support Vector Machine (SVM)

Type: Margin-based classifier
How it works:
SVM finds the optimal hyperplane that maximizes the margin between survivors and non-survivors.

Why used:

Powerful for high-dimensional data

Can use different kernels for non-linear boundaries

6️⃣ Linear Regression (as classifier)

Type: Regression converted to classification
How it works:
Predicts continuous values and converts them to 0 or 1 using a threshold.

Why used:

Baseline comparison

Shows why classifiers are better for classification tasks

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC AUC

🏆 Best Results
Model	Validation AUC	Test AUC
Logistic Regression	0.86	0.86
KNN	0.85	0.85
Decision Tree	0.86	0.84
⚙️ Hyperparameter Tuning

Used GridSearchCV (5-fold cross validation):

KNN: n_neighbors, weights

Logistic Regression: C, solver

SVM: C, kernel, gamma

Decision Tree: max_depth, min_samples_split

Naive Bayes: var_smoothing

📊 Visualizations

Survival Distribution

Correlation Heatmap

ROC Curves (Validation & Test)


