import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('creditcard.csv')  # Replace with your dataset path

# Preprocess the data
X = df.drop(['Class'], axis=1)  # Features
y = df['Class']  # Target (0: Genuine, 1: Fraudulent)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=5)
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier(random_state=42)

# Train and evaluate KNN
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Classifier:")
print(classification_report(y_test, knn_pred))
print("Accuracy:", accuracy_score(y_test, knn_pred))

# Train and evaluate Naive Bayes
naive_bayes.fit(X_train, y_train)
nb_pred = naive_bayes.predict(X_test)
print("\nNaive Bayes Classifier:")
print(classification_report(y_test, nb_pred))
print("Accuracy:", accuracy_score(y_test, nb_pred))

# Train and evaluate Decision Tree
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(X_test)
print("\nDecision Tree Classifier:")
print(classification_report(y_test, dt_pred))
print("Accuracy:", accuracy_score(y_test, dt_pred))

# Visualize results
import matplotlib.pyplot as plt

models = ['KNN', 'Naive Bayes', 'Decision Tree']
accuracies = [
    accuracy_score(y_test, knn_pred),
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, dt_pred)
]

plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
