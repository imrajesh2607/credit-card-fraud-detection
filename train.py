import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

import warnings
warnings.filterwarnings("ignore")

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

"""
Under-Sampling

Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

Number of Fraudulent Transactions --> 492
"""

legit_sample = legit.sample(n=492)

"""Concatenating two DataFrames"""

new_dataset = pd.concat([legit_sample, fraud], axis=0)

"""Splitting the data into Features & Targets"""

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

"""Split the data into Training data & Testing Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

"""
Model Training using Logistic Regression
"""

model = LogisticRegression()

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

"""
Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(f'Accuracy on Training data : {training_data_accuracy}')

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print(f'Accuracy score on Test Data : {test_data_accuracy}')

filename = 'trainedModel.pth'
pickle.dump(model, open(filename, 'wb'))

print('-'*65)
print(f'Model Trained Succesfully and is saved to {filename}')
print('-'*65)

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(f'Overall Accuracy: {result*100} %')