"""
This code is a Python implementation of a data analysis workflow using the Abalone dataset from the UCI Machine Learning Repository. The code performs the following steps:
1. Fetches the Abalone dataset using the `fetch_ucirepo` function.
2. Preprocesses the data by one-hot encoding the 'Sex' variable.
3. Combines the features and target variable into a single DataFrame for analysis.
4. Visualizes the distribution of the target variable 'Rings' and the correlation matrix of the features.
5. Removes outliers using the Interquartile Range (IQR) method and transforms the
6. Creates scatter plots with linear regression lines to explore the relationships between 'Diameter', 'Shell
7. Draw conclusions about the suitability of linear regression for this dataset based on the visualizations and correlations observed.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.encoding import OneHotEncoder


X = abalone.data.features
y = abalone.data.targets

# One Hot Encode Sex
ohe = OneHotEncoder(variables=['Sex'])
X = ohe.fit_transform(X)

# View
df = pd.concat([X,y], axis=1)

#print(df.head())

# plt.hist(y)
# plt.title('Rings [Target Variable] Distribution');
# #plt.show()

# corr_matrix = df.drop(['Sex_M', 'Sex_I', 'Sex_F'], axis=1).corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# #plt.show()

# sns.pairplot(df)
# plt.show()


#Removing the outliers and transforming the target variable to logarithms should result 
# in the next plot of the pairs.

# Remove outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)].copy()

# Transform target variable to logarithm (ensure positive values)
df_no_outliers['Rings'] = np.log(df_no_outliers['Rings'] + 1)

sns.lmplot(x="Diameter", y="Rings", hue="Sex_M", col="Sex_F", order=2, data=df_no_outliers)
sns.lmplot(x="Shell_weight", y="Rings", hue="Sex_M", col="Sex_F", order=2, data=df_no_outliers)
plt.show()

# We fetched and learned this dataset from the UCI Machine Learning Repository, which is a popular 
# Is not appropriate for linear regession lol yet will proceed with linear regression. 

