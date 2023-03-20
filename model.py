import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from landmarks import landmarks

# Load the data from CSV file
data = pd.read_csv("C:/Users/vinhh/Project Code/Rep Counter/landmark coordinates.csv", encoding='utf-8')

# Drop any rows with missing values
data.dropna(inplace=True)

# Split the data into features (X) and labels (y)
X = data.drop('stage_position', axis=1)
y = data['stage_position']


# Add feature names to the X dataframe
X.columns = landmarks

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
