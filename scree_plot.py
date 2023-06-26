import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # principal component analysis
from sklearn.preprocessing import StandardScaler # scales data to standard normal distribution

# Import CSV file as a Pandas DataFrame 
heating_data = pd.read_csv('Heating-data.csv', delimiter='\t', index_col='Date') # loads CSV to Pandas

# check for any missing values
heating_data.isnull().sum()

# preprocess them for scikit-learn:
features = ["Sunshine duration [h/day]", "Outdoor temperature [°C]", "Solar yield [kWh/day]", "Solar pump [h/day]", "Valve [h/day]"]

# Separate the target variable 'y' (gas consumption) from the other variables 'x'
target = "Gas consumption [kWh/day]" #dependent variable

x = np.c_[heating_data[features]] # extracts feature values as a matrix
y = np.c_[heating_data[target]] # extracts target values as a one-column matrix

# Scaling the data
model1 = StandardScaler()
model1 = model1.fit(x)
x_scaled = model1.transform(x) # compute and store the scaled data

# Perform PCA on the 'x' variables
model = PCA() # enable all possible principal components
model = model.fit(x_scaled, y)

# Scree plot
print("Explained variance ratio:", model.explained_variance_ratio_)
plt.figure(figsize=(6, 4))
pc_values = np.arange(model.n_components_) + 1
plt.plot(pc_values, model.explained_variance_ratio_, 'o-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.xticks(pc_values)
plt.tight_layout()
plt.show()
