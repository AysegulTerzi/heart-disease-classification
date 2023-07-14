import numpy as np 
import sklearn 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm


df = pd.read_csv("heart.csv", encoding='utf-8')
df = df.dropna()
df['Sex'] = df['Sex'].replace(['F'],0)
df['Sex'] = df['Sex'].replace(['M'],1)
df['ChestPainType'] = df['ChestPainType'].replace(['TA'],0)
df['ChestPainType'] = df['ChestPainType'].replace(['ATA'],1)
df['ChestPainType'] = df['ChestPainType'].replace(['NAP'],2)
df['ChestPainType'] = df['ChestPainType'].replace(['ASY'],3)
df['RestingECG'] = df['RestingECG'].replace(['Normal'],0)
df['RestingECG'] = df['RestingECG'].replace(['ST'],1)
df['RestingECG'] = df['RestingECG'].replace(['LVH'],2)
df['ExerciseAngina'] = df['ExerciseAngina'].replace(['Y'],1)
df['ExerciseAngina'] = df['ExerciseAngina'].replace(['N'],0)
df['ST_Slope'] = df['ST_Slope'].replace(['Down'],0)
df['ST_Slope'] = df['ST_Slope'].replace(['Flat'],1)
df['ST_Slope'] = df['ST_Slope'].replace(['Up'],1)


features = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']]

label = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(features, label,
                                                    test_size=0.30, random_state=46)

## Linear Regression Model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)
linear_predictions = linear_regression_model.predict(X_test) 
linear_mse = sklearn.metrics.mean_squared_error(y_test, linear_predictions)  
linear_rmse = np.sqrt(linear_mse) 

## Neural Network Model
neural_network_model = MLPRegressor(max_iter=1000, random_state=46) 
neural_network_model.fit(X_train, y_train)
nn_predictions = neural_network_model.predict(X_test) 
nn_mse = sklearn.metrics.mean_squared_error(y_test, nn_predictions) 
nn_rmse=np.sqrt(nn_mse)

## Logistic Regression Model
logistic_regression_model = LogisticRegression(random_state=0,solver='lbfgs', max_iter=3000)
logistic_regression_model.fit(X_train, y_train)
logistic_predictions = logistic_regression_model.predict(X_test)
logistic_acc = accuracy_score(y_test, logistic_predictions)

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model = decision_tree_model.fit(X_train,y_train)
decision_tree_predictions = decision_tree_model.predict(X_test)
d_acc=accuracy_score(y_test, decision_tree_predictions)

# SVM Model
SVM_model = svm.SVC(kernel='linear', C=1.0)
SVM_model = SVM_model.fit(X_train,y_train)
SVM_predictions = SVM_model.predict(X_test)
svm_acc=accuracy_score(y_test, SVM_predictions)

print('Linear model RMSE:', linear_rmse)
print('Neural Network model RMSE:', nn_rmse)
print("Logistic Regression accuracy:", logistic_acc)
print("Decision tree Accuracy:",d_acc)
print("Support Vector Machine Accuracy:",svm_acc)