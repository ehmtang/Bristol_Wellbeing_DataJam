# -*- coding: utf-8 -*-
"""
Absenteeism in the workplace

Created on Sat Jan 25 14:11:20 2020

@author: EHMTang
"""

# %%
 
# Import relevant packages
import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from datetime import date, timedelta


# Import statements required for Plotly 
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly import tools
py.init_notebook_mode(connected=True)


# Import sklearn machine learning models
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (make_moons,
                              make_circles,
                              make_classification,
                              make_blobs,
                              make_checkerboard)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier, 
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              BaggingClassifier)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# %%

# Close all opened plots
plt.close("all")

# Set seaborn style
sns.set()

# %%

# Get current working directory
mypath = os.getcwd()

# Import dataset
raw = pd.read_csv(mypath + r"\Staff_absence.csv")
data = pd.read_csv(mypath + r"\Staff_absence.csv")

# %%

# Checking for NULL values and dtypes
print('*' * 100)
print(data.info())
print('*' * 100)
print(data.head(5))
print('*' * 100)
print(data.tail(5))
print('*' * 100)
print("Found anomalous end date value 12/31/4712")
print('*' * 100)


# %%

# Cleaning and assigning types

# Remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Assign to date types
data['Absence Start Date'] = pd.to_datetime(data['Absence Start Date'], errors='coerce')
data['Absence End Date'] = pd.to_datetime(data['Absence End Date'], errors='coerce')

# Drop NaN values
data = data.dropna() # 33 NaT in data set

# %% 

# Assign start and end dates

earliest_absence = min(data["Absence Start Date"])
last_absence = max(data["Absence End Date"])


datelist = pd.date_range(start = earliest_absence,
                         end = last_absence)


# %%

# Randomise dataset
data_randomised = shuffle(data)
data_randomised.reset_index()

training_range = round(0.7 * len(data))

data_train = data_randomised.iloc[0 : training_range]
data_test = data_randomised.iloc[training_range : len(data)]

data_train.reset_index()
data_test.reset_index()

# %%

# Preliminary Data Visualisation

# Scatter Matrix
from plotly.offline import plot

fig = px.scatter_matrix(data_train,
                        dimensions=["FTE",
                                    "FTE Days Lost",
                                    "Calendar Days Lost",
                                    "Total FTE Calendar Days "],
                        )

fig.update_traces(diagonal_visible=True)
plot(fig)

# Histograms
fig = px.histogram(data_train, x="FTE")
plot(fig)

fig = px.histogram(data_train, x="FTE Days Lost")
plot(fig)

fig = px.histogram(data_train, x="Calendar Days Lost")
plot(fig)

fig = px.histogram(data_train, x="Total FTE Calendar Days ")
plot(fig)

fig = px.histogram(data_train, x="First Day Absent").update_xaxes(categoryorder="total descending")
plot(fig)

# %% 

# Create new group: FTE_group

data_train["FTE_group"] = ""
data_train["less_0.6"] = data_train["FTE"] < 0.6
data_train["0.6 to 0.8"] = (0.6 <= data_train["FTE"]) & (data_train["FTE"] < 0.8)
data_train["0.8 to 1.0"] = (0.8 <= data_train["FTE"]) & (data_train ["FTE"] < 1.0)
data_train["1.0"] = data_train["FTE"] == 1

for i in range(len(data_train)):
    
    if data_train["less_0.6"][i] == True:
    
        data_train["FTE_group"][i] = "less than 0.6"

"""
    elif data_train["0.6 to 0.8"][i] == True:
        data_train["FTE_group"][i] = "from 0.6 to 0.8"
    
    elif data_train["0.6 to 0.8"][i] == True:
        data_train["FTE_group"][i] = "from 0.6 to 0.8"

    elif data_train["0.8 to 1.0"][i] == True:
        data_train["FTE_group"][i] = "from 0.8 to 1.0"

    elif data_train["1.0"][i] == True:
        data_train["FTE_group"][i] = "equal to 1.0"
"""
# %%
