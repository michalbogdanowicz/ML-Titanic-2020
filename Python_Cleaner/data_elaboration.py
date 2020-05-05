#!/usr/bin/python3

# taken from : https://www.kaggle.com/schmitzi/cleaning-titanic-data-and-running-scikitlearn
import pandas as pd
import numpy as np

data = pd.read_csv("titanic.csv", sep=",", header=0, index_col=0)

print("Data structure:")
print("***************")
print(data.columns)
print(data.dtypes)
print("\nExample:")
print("**********")
print(data.head())
print("\nStatistics:")
print("*************")
print(data.describe())
print("Correlations:")
print(data.corr())
print("*************")
print("Columns with <10 categories:")
for i in data.columns:
    catdat = pd.Categorical(data[i])
    if len(catdat.categories)>9:
        continue

    print(i," ",pd.Categorical(data[i]))

print(data.columns)

data.age.fillna(value=data.Age.mean(), inplace=True)
data.fare.fillna(value=data.Fare.mean(), inplace=True)
data.embarked.fillna(value=(data.Embarked.value_counts().idxmax()), inplace=True)

print("Extracting titles and adding column...")
titles = pd.DataFrame(data.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1), columns=["Title"])
print(pd.Categorical(titles.Title))
data = data.join(titles)

print("Calculating family size and adding column...")
fsiz = pd.DataFrame(data.apply(lambda x: x.SibSp+x.Parch, axis=1), columns=["FSize"])
data = data.join(fsiz)

# drop useless columns
data.drop('Name', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Body', axis=1, inplace=True)

# no need for the following as the sum is used
data.drop('Parch', axis=1, inplace=True)
data.drop('SibSp', axis=1, inplace=True)

# generate numerical output
print("Conveting to numerical output...")

for col in data.select_dtypes(exclude=["number"]).columns:
    print("Converting column "+col+"...")
    data[col] = data[col].astype('category')
    print(data[col].cat.categories)
    data[col] = data[col].cat.codes

data.to_csv("titanic_clean.csv")
