"""
================================
Exploratory Data Analysis
================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%

# Load the dataset
file_path = '../data/Copy of master_sheet_QDS_CeZn.xlsx'
df = pd.read_excel(file_path)
# %%

# Display the shape of the dataset (number of rows and columns)
print("Shape of the Dataset:", df.shape)
# %%

# Display summary statistics (mean, median, standard deviation, etc.)
print("\nSummary Statistics:\n", df.describe())
# %%

# Identify and count missing values in each column
print("\nMissing Values:\n", df.isnull().sum())
# %%

# Selecting key numerical columns for distribution analysis
numerical_columns = ['PMS_concentration g/L', 'Co (intial content of DS pollutant)',
                     'MO_conc_mg/L', 'NP_conc_mg/L', 'NX_conc_mg/L',
                     'TC_conc_mg/L', 'IBU_conc_mg/L', 'catalyst dosage_g/L', 'pH',
                     'removal%', 'K Reaction rate constant (k 10-2min-1)', 'Ct']
# %%

" " " Histogram " " "
# Plotting histograms for the selected columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
# %%


" " " Box Plot " " "
# Plotting box plots for the same variables to identify outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
# %%


" " " Scatter Plot " " "
# Selecting pairs of variables for scatter plots
scatter_pairs = [('PMS_concentration g/L', 'removal%'),
                 ('Co (intial content of DS pollutant)', 'removal%'),
                 ('catalyst dosage_g/L', 'removal%'),
                 ('pH', 'removal%')]

# Plotting scatter plots for selected pairs of variables
plt.figure(figsize=(15, 8))
for i, (x, y) in enumerate(scatter_pairs, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=df[x], y=df[y])
    plt.title(f'{x} vs {y}')
plt.tight_layout()
plt.show()
# %%

" " " Correlation Matrix " " "
# Calculating and plotting the correlation matrix
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# %%

" " " Line Chart " " "
# Plotting line charts for selected variables over time or a continuous variable
time_variable = 'time_min'  # Replace with the actual time column in your dataset
line_chart_columns = ['PMS_concentration g/L', 'Co (intial content of DS pollutant)', 'MO_conc_mg/L',
                      'NP_conc_mg/L', 'NX_conc_mg/L', 'TC_conc_mg/L', 'IBU_conc_mg/L',
                      'catalyst dosage_g/L', 'pH', 'removal%', 'K Reaction rate constant (k 10-2min-1)', 'Ct']

plt.figure(figsize=(15, 8))
for i, col in enumerate(line_chart_columns, 1):
    plt.subplot(3, 4, i)  # Adjust the subplot parameters based on the number of columns
    sns.lineplot(x=df[time_variable], y=df[col])
    plt.title(f'{col} over time')
plt.tight_layout()
plt.show()
# %%

" " " PyPlot " " "
# Selecting categorical columns for visualization
categorical_columns = ['system']  # Add more columns as needed

# Plotting bar charts for selected categorical columns
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, len(categorical_columns), i)
    sns.countplot(x=df[col])
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()