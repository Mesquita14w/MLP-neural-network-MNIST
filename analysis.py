import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
This file it's just a playground (very simple, 
because the data already cleaned and organized in kaggle) 
that i use to explore the dataset and to investigate the data 
and the features. I will use this file to create some plots 
and to analyze the data. I mentioned early, i don't like to use 
jupyter notebooks, so i will use this file to do the data analysis 
step by step. I will gonna write all the commands i use to explore 
the data and to create some plots. But i will not write all the 
code here, i execute the commands step by step and understanding 
all data by myself, without jupyter notebooks.
"""

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Overview of the data using display:
display(train.head())
display(test.head())

# Plot the digit distribution of the train dataset:
plt.figure(figsize=(10, 5))
sns.countplot(data=train, x="label")
plt.title("Distribuição dos dígitos")
plt.xlabel("Dígito")
plt.ylabel("Quantidade")
plt.show()

# Missing values in the two datasets:
print(f"Valores ausentes no treino: {train.isnull().sum().sum()}")
print(f"Valores ausentes no teste: {test.isnull().sum().sum()}")