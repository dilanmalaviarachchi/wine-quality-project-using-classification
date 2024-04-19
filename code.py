#importing libraries 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#load wine quality dataset 
data = pd.read_csv("C:\\Users\\Malavi\\Desktop\\australia\\wine_quality.csv.crdownload")

#spliting data into features (x) and (y)
X = data.drop('quality', axis=1)
y = data['quality']

#define different sets of features 

features_list = [
    ['fixed acidity'],
    ['volatile acidity'],
    ['citric acid'],
    ['residual sugar'],
    ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides'],
    list(X.columns)  # All features
]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize dictionaries to store testing accuracies for logistic regression and SVM
lr_accuracies = {}
svm_accuracies = {}

# Train and evaluate classifiers for each set of features
for features in features_list:
    # Subset features
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    # Train logistic regression classifier
    lr_classifier = LogisticRegression(max_iter=10000)
    lr_classifier.fit(X_train_subset, y_train)
    
    # Train SVM classifier
    svm_classifier = SVC()
    svm_classifier.fit(X_train_subset, y_train)
    
    # Predict on the testing set
    lr_predictions = lr_classifier.predict(X_test_subset)
    svm_predictions = svm_classifier.predict(X_test_subset)
    
    # Calculate testing accuracies
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    
    # Store accuracies in dictionaries
    lr_accuracies['+'.join(features)] = lr_accuracy
    svm_accuracies['+'.join(features)] = svm_accuracy

# Print testing accuracies
print("Features used\t\t\tLogistic Regression\t\tSupport Vector Machine")
for features in features_list:
    features_str = '+'.join(features)
    print(f"{features_str:<30}\t\t{lr_accuracies.get(features_str):.4f}\t\t\t\t\t\t{svm_accuracies.get(features_str):.4f}")
