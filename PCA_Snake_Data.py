""" 
PCA_Snake_Data.py
Usage: Press play

This file takes the complete snake dataset 'complete_snake_dataset.csv' and uses it to make a PCA plot.
This plot either colours the point blue (if neuro) or red (if hemo). The clusters are created, with red points being more clustered together and 
blue points being more clustered together. 

This PCA plot is used to show how a SVM algorithm may work, by creating a hyperplane seperating different classes of values.
The algorithm then makes new predictions on new input variables, by which side of the hyperplane they are on.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset and read it. Put it in a object called data
data = pd.read_csv("complete_snake_dataset.csv")


# Because my dataset target variables (venom type, either neuro or hemo) are not numeric, I have to convert them
#hemo == 1 and neuro == 0 for these purposes
data['VT'] = data['VT'].map({'Hemo': 1, 'Neuro': 0}) #.map({'Hemo': 1, 'Neuro': 0}): This method applies a mapping to the values in the 'VT' column. It replaces each value according to the provided dictionary. In this case:


# Separate evidence variables (different compositions for the 10+ components) from the target variable (venom type)
# This accesses the rows and columns of the DataFrame data using integer-based indexing (iloc).
""" 
: indicates that we're selecting all rows.
1:-1 indicates that we're selecting columns starting from index 1 (exclusive) up to the second-to-last column (exclusive). 
This typically selects all columns except for the first and last columns, assuming the DataFrame data has features (evidence variables) in columns 1 to 
second-to-last and the target variable in the last column.

"""
X = data.iloc[:, 1:15]  # evidence variables, selecting the rows 1 - 14 (the proteins)
y = data.iloc[:, -1]    # This line of code selects the last column of the DataFrame data as the target variable (y - Venom Type) using integer-location based indexing. -1 refers to the last column in the DataFrame.


# These two lines of code scale the evidence variables in my dataset
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) #X_scaled contains the scaled features (evidence variables) of your dataset

# Perform PCA
pca = PCA(n_components=2) # creates an instance of the PCA class, specifying that you want to reduce the dimensionality of your data to two principal components.
X_pca = pca.fit_transform(X_scaled) # fits the PCA model to the scaled features in X_scaled and then applies the dimensionality reduction to transform the data into the principal component space.

# Plot PCA
""" 
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='Neurotoxic'): 
This plots the data points corresponding to the 'Neurotoxic' class in the PCA space. 
It selects the first principal component for x-axis (index 0) and the second principal component 
for y-axis (index 1) for those instances where y is equal to 0 (indicating the 'Neurotoxic' class). 
These points are colored blue and labeled as 'Neurotoxic'.

plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='Hemotoxic'): 
This plots the data points corresponding to the 'Hemotoxic' class in the PCA space. Similarly, 
it selects the first principal component for x-axis and the second principal component for y-axis 
for those instances where y is equal to 1 (indicating the 'Hemotoxic' class). 
These points are colored red and labeled as 'Hemotoxic'.
"""
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='Neurotoxic')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='Hemotoxic')

# Add explained variance ratio to the plot
# These lines of code add text annotations to the PCA plot, indicating the percentage of variance explained by each principal component:
# plt.text(X_pca[:, 0].max(), X_pca[:, 1].max(), f'PC1: {pca.explained_variance_ratio_[0]*100:.2f}%', fontsize=10, ha='right')
# plt.text(X_pca[:, 0].min(), X_pca[:, 1].min(), f'PC2: {pca.explained_variance_ratio_[1]*100:.2f}%', fontsize=10, ha='left')

plt.title('PCA Plot of Venom Types')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper left') #this moves the legend to the top left corner
plt.show()
