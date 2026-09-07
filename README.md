# linear-SVM-for-iris-classification

Overview

This project demonstrates how to build and evaluate a Linear Support Vector Machine (SVM) classifier using Python and the built-in Iris dataset from scikit-learn.

The model performs binary classification by distinguishing Iris Setosa flowers from the other two Iris species. The classification uses two features:

Petal Length
Petal Width

The features are standardized before training, and the trained SVM is used to calculate several classification metrics and visualize the decision boundary, margins, and support vectors.

Features
Loads the Iris dataset using scikit-learn
Uses petal length and petal width as input features
Converts the original three-class problem into a binary classification problem
Splits the data into training and testing sets
Standardizes features using StandardScaler
Trains a linear SVM classifier
Calculates:
Accuracy
Precision
Recall
F1-score
Classification report
Displays the SVM weight vector and bias
Identifies support vectors
Calculates the SVM margin width
Visualizes:
Training data
Decision boundary
SVM margins
Support vectors
Technologies Used
Python 3
NumPy
Pandas
Matplotlib
Scikit-learn
Dataset

The project uses the Iris dataset, which contains 150 samples of Iris flowers from three species:

Iris Setosa
Iris Versicolor
Iris Virginica

Each sample contains four measurements:

Sepal Length
Sepal Width
Petal Length
Petal Width

This project only uses:

Petal Length
Petal Width


These are selected using:

X = iris.data[:, [2, 3]]

Binary Classification

The original Iris dataset contains three classes. This project converts it into a binary classification problem.

y_binary = np.where(y == 0, 1, 0)


The labels are therefore:

Label	Class
1	Setosa
0	Non-Setosa

The Non-Setosa class combines Versicolor and Virginica.

Project Workflow

The machine learning pipeline follows these steps:

Load Iris Dataset
       ↓
Select Petal Length & Petal Width
       ↓
Convert to Binary Classification
       ↓
Train/Test Split
       ↓
Feature Standardization
       ↓
Train Linear SVM
       ↓
Make Predictions
       ↓
Evaluate Model
       ↓
Calculate Decision Boundary & Margins
       ↓
Visualize Results

Installation

Make sure Python 3 is installed on your system.

Install the required libraries using:

pip install numpy pandas matplotlib scikit-learn


Alternatively, create a requirements.txt file:

numpy
pandas
matplotlib
scikit-learn


Then install the dependencies with:

pip install -r requirements.txt

Running the Project

Save the Python code in a file such as:

svm_iris.py


Run it using:

python svm_iris.py


The program will print the dataset information, class distribution, model predictions, evaluation metrics, SVM parameters, support vectors, and margin width.

A plot showing the decision boundary and margins will also be displayed.

Model Configuration

The project uses a Support Vector Classifier with a linear kernel:

svm_model = SVC(kernel="linear", C=1.0)

Parameters

Kernel: linear

A linear kernel is used because the project focuses on understanding the linear SVM decision boundary.

C: 1.0

The C parameter controls the trade-off between maximizing the margin and minimizing classification errors.

Feature Standardization

Before training, the input features are standardized:

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


Standardization transforms the features so that they have approximately:

Mean = 0
Standard deviation = 1


This is particularly useful for SVM models because the scale of the input features can influence the resulting decision boundary.

Train/Test Split

The dataset is divided into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)


The configuration uses:

80% training data
20% testing data
random_state=42 for reproducibility
stratify=y_binary to preserve the class distribution
Evaluation Metrics

The following metrics are calculated:

Accuracy

Accuracy represents the proportion of predictions that are correct.

accuracy_score(y_test, y_pred)

Precision

Precision measures how many samples predicted as a particular class are actually members of that class.

precision_score(y_test, y_pred)

Recall

Recall measures how many actual positive samples were correctly identified.

recall_score(y_test, y_pred)

F1-Score

The F1-score combines precision and recall into a single metric.

f1_score(y_test, y_pred)

Classification Report

A complete classification report is also generated:

classification_report(
    y_test,
    y_pred,
    target_names=["Non-Setosa", "Setosa"]
)

SVM Decision Boundary

After training, the model's weight vector and bias are extracted:

w = svm_model.coef_[0]
b = svm_model.intercept_[0]


For a linear SVM, the decision boundary is represented by:

w₁x₁ + w₂x₂ + b = 0


The code calculates the corresponding boundary line using:

decision_boundary = (-(w[0] * xx + b) / w[1])

SVM Margins

The two margin boundaries are calculated using:

margin_positive = (-(w[0] * xx + b - 1) / w[1])

margin_negative = (-(w[0] * xx + b + 1) / w[1])


These represent the boundaries:

wᵀx + b = +1
wᵀx + b = -1


The distance between these two margin boundaries represents the width of the SVM margin.

Support Vectors

Support vectors are the training samples that define the position of the SVM decision boundary and margin.

They are obtained using:

support_vectors = svm_model.support_vectors_


The program prints the number and standardized coordinates of the support vectors.

The support vectors are also highlighted in the visualization using circles around the corresponding points.

Margin Width

The width of the SVM margin is calculated using:

margin_width = 2 / np.linalg.norm(w)


For a linear SVM, the margin width is:

2 / ||w||


where ||w|| is the Euclidean norm of the weight vector.

Visualization

The program generates a plot containing:

Setosa training samples
Non-Setosa training samples
Decision boundary
Positive margin
Negative margin
Support vectors

The axes represent standardized petal length and petal width.

The visualization helps demonstrate how the linear SVM separates Setosa flowers from the remaining Iris classes.

Expected Output

The console output will contain information similar to:

Feature names:
['sepal length (cm)' 'sepal width (cm)'
 'petal length (cm)' 'petal width (cm)']

Dataset shape:
(150, 2)

Class distribution:
0    100
1     50

Training samples: 120
Testing samples: 30

Predicted labels:
[...]

Accuracy: ...
Precision: ...
Recall: ...
F1-score: ...

Classification Report:
...

Weight Vector:
[...]

Bias:
...

Number of Support Vectors: ...

Support Vectors:
[...]

Margin Width: ...


The exact metric values and support vectors are generated when the program is executed.

Project Structure

A simple project structure can be:

linear-svm-iris/
│
├── svm_iris.py
├── README.md
└── requirements.txt

Important Notes

The code performs binary classification, not three-class Iris classification.

Specifically:

Setosa       → 1
Versicolor   → 0
Virginica    → 0


Therefore, the model learns to distinguish Setosa from all non-Setosa samples.

Also, the decision boundary and margins are plotted using the standardized feature values, not the original centimeter measurements.

Possible Improvements

This project can be extended by:

Comparing linear SVM with RBF and polynomial kernels
Testing different values of C
Adding a confusion matrix
Calculating ROC-AUC
Visualizing the test set separately
Comparing SVM with Logistic Regression, KNN, and Decision Trees
Performing cross-validation
Adding command-line arguments for model parameters
Saving the trained model using joblib
License

This project is intended for educational and demonstration purposes.
