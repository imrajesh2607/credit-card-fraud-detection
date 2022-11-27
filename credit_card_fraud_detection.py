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

# first 5 rows of the dataset
credit_card_data.head()

credit_card_data.tail()

# dataset informations
credit_card_data.info()

# checking the number of missing values in each column
credit_card_data.isnull().sum()

# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()

"""
This Dataset is highly unblanced

0 --> Normal Transaction

1 --> fraudulent transaction
"""

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# print(legit.shape)
# print(fraud.shape)

# statistical measures of the data
legit.Amount.describe()

fraud.Amount.describe()

# compare the values for both transactions
credit_card_data.groupby('Class').mean()

"""
Under-Sampling

Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

Number of Fraudulent Transactions --> 492
"""

legit_sample = legit.sample(n=492)

"""Concatenating two DataFrames"""

new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

"""Splitting the data into Features & Targets"""

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# print(X)

# print(Y)

"""Split the data into Training data & Testing Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)

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

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

df_col = [i for i in X_test.columns]
print(df_col)
test = pd.DataFrame(columns = df_col)
test = X_test.iloc[0]
x = [j for j in test]
print(f'Rows\n: {x}')
prediction = model.predict([test])
print(prediction)

filename = 'trainedModel.pth'
pickle.dump(model, open(filename, 'wb'))
print(f'Model Trained Succesfully and is saved to {filename}')

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

df = pd.DataFrame(columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'])
lst = [0, -1.359807134, -0.072781173, 2.536346738, 1.378155224, -0.33832077, 0.462387778, 0.239598554, 0.098697901, 0.36378697, 0.090794172, -0.551599533, -0.617800856, -0.991389847, -0.311169354, 1.468176972, -0.470400525, 0.207971242, 0.02579058, 0.40399296, 0.251412098, -0.018306778, 0.277837576, -0.11047391, 0.066928075, 0.128539358, -0.189114844, 0.133558377, -0.021053053, 149.62]
# time = request.form.get("Time")
# lst.append(time)
# for i in range(1,29):
#       s = 'V'+str(i)
#       x = request.form.get(s)
#       lst.append(x)
# amount = request.form.get("Amount")
# lst.append(amount)
df.loc[len(df)] = lst
filename = 'trainedModel.pth'
loaded_model = pickle.load(open(filename, 'rb'))
prediction = loaded_model.predict(df)
if prediction[0]:
      print('It is Fraud')
else:
      print('Its not Frauud')