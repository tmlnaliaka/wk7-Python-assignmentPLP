import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean dataset (handle missing values)
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean(numeric_only=True))
        print("\nMissing values filled with mean.")
    else:
        print("\nNo missing values found.")

except Exception as e:
    print(f"Error loading or processing dataset: {e}")
    exit()

# Task 2: Basic Data Analysis
try:
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Group by species and compute mean for numerical columns
    grouped_means = df.groupby('species').mean(numeric_only=True)
    print("\nMean values by species:")
    print(grouped_means)

    # Observations
    print("\nObservations:")
    print("- Setosa has the smallest mean petal length and width.")
    print("- Versicolor has intermediate values for most features.")
    print("- Virginica has the largest mean sepal and petal measurements.")

except Exception as e:
    print(f"Error during data analysis: {e}")

# Task 3: Data Visualization
try:
    # Visualization 1: Line chart of mean feature values by species
    plt.figure(figsize=(10, 6))
    for column in df.columns[:-1]:  # Exclude species column
        plt.plot(grouped_means.index, grouped_means[column], marker='o', label=column)
    plt.title('Mean Feature Values by Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Mean Value (cm)')
    plt.legend()
    plt.savefig('mean_features_line.png')
    plt.close()

    # Visualization 2: Bar chart of mean petal length by species
    plt.figure(figsize=(8, 6))
    plt.bar(grouped_means.index, grouped_means['petal length (cm)'])
    plt.title('Average Petal Length by Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.savefig('petal_length_bar.png')
    plt.close()

    # Visualization 3: Histogram of sepal length
    plt.figure(figsize=(8, 6))
    plt.hist(df['sepal length (cm)'], bins=20, edgecolor='black')
    plt.title('Distribution of Sepal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.savefig('sepal_length_histogram.png')
    plt.close()

    # Visualization 4: Scatter plot of sepal length vs petal length
    plt.figure(figsize=(8, 6))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                   label=species, alpha=0.6)
    plt.title('Sepal Length vs Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.savefig('sepal_petal_scatter.png')
    plt.close()

    print("\nVisualizations saved as PNG files:")
    print("- mean_features_line.png")
    print("- petal_length_bar.png")
    print("- sepal_length_histogram.png")
    print("- sepal_petal_scatter.png")

except Exception as e:
    print(f"Error creating visualizations: {e}")

print("\nAnalysis complete. Check the generated PNG files for visualizations.")