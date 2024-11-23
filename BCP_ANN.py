# Importing Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow.keras as tf

# Uploading the CSV File
data = pd.read_csv(r"C:\Users\dasad\PycharmProjects\Breast_Cancer_Prediction\breast_cancer.csv")
# print(data.info())

# Data Preprocessing
#Removing Unnecessary Columns
data = data.drop(['id'], axis = 1)

#Label Data
encoder = LabelEncoder()
data['diagnosis'] = encoder.fit_transform(data['diagnosis'])
# print(data.head())
# print(data.info())

#Feature as Input and Output
x = data.drop(['diagnosis'],axis=1)
y = data['diagnosis']
allcolumns = x.columns

#Normalization of Feature Values
sc = StandardScaler()
x = sc.fit_transform(x)

#Split the Dataset into Train and Test
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.8,random_state=0, shuffle=True)

# Creating the Artificial Neural Network
# print(x.shape)
model = tf.models.Sequential()

#Add Layers
# Input Layer
model.add(tf.layers.Dense(16, input_dim=30, activation = 'relu'))
# Hidden Layer
model.add(tf.layers.Dense(32, activation = 'relu'))
# Output Layer
model.add(tf.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ["accuracy"])

# Fitting the Model
model.fit(xtrain,ytrain,epochs=100)

#Predicting the Output
ypred = model.predict(xtest)
ypred = ypred.round()

# Calculating the Metrics
cm = confusion_matrix(ytest, ypred)
acc = accuracy_score(ytest, ypred)

print("The Confusion Matrix is: \n", cm)
print("The Accuracy of the Model is: ", acc)

#Custom/User Input Data
# data_array = np.zeros(30)
#print("\n\n\t\tEnter Custom Data To Predict:\n\t\t(Enter all Data Carefully)\n")
#for index, column in enumerate(allcolumns):
#    user_input = float(input("\t" + column + ": "))
#    data_array[index] = user_input

data_array = np.array([9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773])
data_array = data_array.reshape(1, -1)
data_array = sc.transform(data_array)
prediction = model.predict(data_array)

if prediction[0][0] >= 0.5:
    prediction = 'MALIGNANT'
else:
    prediction = 'BENIGN'

print("Based on the Data you have entered, the System has predicted that the Cancer is in "+prediction+" Stage")