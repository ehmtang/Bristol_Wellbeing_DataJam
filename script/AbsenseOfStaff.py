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
import datetime
from matplotlib.colors import ListedColormap


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
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

data['Absence Start Date'] = data['Absence Start Date'].astype('datetime64[ns]')


# %%

# Randomise dataset
data_randomised = shuffle(data)
data_randomised.reset_index()

training_range = round(0.7 * len(data))

data_train = data_randomised.iloc[0:training_range]
data_test = data_randomised.iloc[training_range + 1 : len(data)]


# %%

# Data Visualisation

# Scatter Matrix
from plotly.offline import plot

fig = px.scatter_matrix(data,
                        dimensions=["FTE",
                                    "FTE Days Lost",
                                    "Calendar Days Lost",
                                    "Total FTE Calendar Days "],
                        
                        color = "First Day Absent")

fig.update_traces(diagonal_visible=True)

plot(fig)
# %%
# Histogram

from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2)

fig.add_trace(
    go.histogram(data, x="FTE"),
    row=1, col=1
    )

fig.add_trace(
    go.histogram(data, x="FTE Days Lost"),
    row=2, col=1
    )

fig.add_trace(
    go.histogram(data, x="Calendar Days Lost"),
    row=1, col=2
    )

fig.add_trace(
    go.histogram(data, x="Total FTE Calendar Days "),
    row=1, col=2
    )

plot(fig)


# Heatmap

# Violin plot


# %% 

# FTE catagoriser

data['FTE_categorise'] = ""
data['less_0.6'] = data['FTE'] < 0.6
data['0.6 to 0.8'] = (0.6 <= data['FTE']) & (data['FTE'] < 0.8)
data['0.8 to 1.0'] = (0.8 <= data['FTE']) & (data ['FTE'] < 1.0)
data['1.0'] = data['FTE'] == 1

# df = df[(df['closing_price'] >= 99) & (df['closing_price'] <= 101)]


# %%








