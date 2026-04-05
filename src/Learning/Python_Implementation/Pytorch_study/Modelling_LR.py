from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from feature_engine.encoding import OneHotEncoder

# fetch dataset
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
X = abalone.data.features
y = abalone.data.targets

# One Hot Encode Sex
ohe = OneHotEncoder(variables=['Sex'])
X = ohe.fit_transform(X)

# Drop Whole Weight and Length (multicolinearity)
X.drop(['Whole_weight', 'Length'], axis=1, inplace=True)

# View
df = pd.concat([X,y], axis=1)

# Let's create a Pipeline to scale the data and find outliers using KNN Classifier
steps = [
('scale', StandardScaler()),
('LOF', LocalOutlierFactor(contamination=0.05))
]
# Fit and predict
outliers = Pipeline(steps).fit_predict(X)

# Add column
df['outliers'] = outliers

# Modeling
df2 = df.query('Height < 0.3 and Rings > 2 and outliers != -1').copy()
X = df2.drop(['Rings', 'outliers'], axis=1)
y = np.log(df2['Rings'])

lr = LinearRegression()
lr.fit(X, y)

predictions = lr.predict(X)

df2['Predictions'] = np.exp(predictions)
print(root_mean_squared_error(df2['Rings'], df2['Predictions']))