import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

s = pd.read_csv("social_media_usage.csv")
print(s.shape)
print(s.dtypes)

def clean_sm(x):
    return np.where(x == 1, 1, 0)

toy = {"Column 1" : [9,1,11],
       "Column 2" : [20,14,1]}
toy_df = pd.DataFrame(toy)
toy_df_2 = toy_df.applymap(clean_sm)
print(toy_df)
print(toy_df_2)

ss_data = {"sm_li" : s["web1h"],
           "income" : s["income"],
           "education" : s["educ2"],
           "parent" : s["par"],
           "married" : s["marital"],
           "female" : s["gender"],
           "age": s["age"]}

ss = pd.DataFrame(ss_data)
print(ss.head)

ss["sm_li"] = ss["sm_li"].apply(clean_sm)
print(ss["sm_li"])

ss["female"] = np.where(ss["female"] == 2, 1, 0)
ss["married"] = np.where(ss["married"] == 1, 1, 0)
ss["parent"] = np.where(ss["parent"] == 1, 1, 0)

print(ss["female"])
print(ss["married"])
print(ss["parent"])

ss["income"] = np.where((ss["income"] > 9), np.nan, ss["income"])
ss["age"] = np.where((ss["age"] > 98), np.nan, ss["age"])
ss["education"] = np.where((ss["education"] > 8), np.nan, ss["education"])

ss.dropna(subset=["income", "age", "education"], inplace=True)

ss.isnull().sum()

print(ss.head())
print(ss.isnull())

sns.pairplot(ss, hue = "sm_li", diag_kind = "kde")
plt.show()

y = ss["sm_li"]

x = ss.drop(columns=["sm_li"])

print(y.head())
print(x.head())

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size= .2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lr_model = LogisticRegression(class_weight="balanced")

lr_model.fit(X_train, Y_train)

Y_pred = lr_model.predict(X_test)

model_accuracy = accuracy_score(Y_test, Y_pred)
print(model_accuracy)

conf_matrix = confusion_matrix(Y_test, Y_pred)
print(conf_matrix)

confusion_df = pd.DataFrame(conf_matrix, columns= ["Actual LinkedIn User", "Actual Non-LinkedIn User"], index=["Predicted LinkedIn User", "Predictive Non-LinkedIn User"])
print(confusion_df)

precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
print(precision)
print(recall)
print(f1_score)

class_report = classification_report(Y_test, Y_pred)
print(class_report)

person1 = pd.DataFrame({
    'income': [8],
    'education': [7],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [42]
})

person2 = pd.DataFrame({
    'income': [8],
    'education': [7],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [82] 
})

probabilities_person1 = lr_model.predict_proba(person1)[:, 1]
probabilities_person2 = lr_model.predict_proba(person2)[:, 1]
print(probabilities_person1)
print(probabilities_person2)
