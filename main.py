import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle




def load_data(csv_path_: str):
    # Load dataset
    data = pd.read_csv(csv_path_)
    # Preview data
    print(data.head())
    return data

def split_data(data: pd.DataFrame):
    # split data to features and label
    X = data.drop('0', axis=1)
    y = data['0']

    return X, y

def show_images(X: pd.DataFrame, y: pd.Series, n_images: int):
    # Display sample images

    # Reshape into 28x28
    X_images = X.values.reshape(-1, 28, 28)

    # Plot the first 10 images
    plt.figure(figsize=(10 , 5))
    for i in range(n_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_images[i], cmap="gray")
        plt.title(f"Label: {y[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def show_classes_info(y: pd.Series):
    # show number of unique classes
    print(f"\nNumber of unique classes: {y.nunique()}")

    # show class distribution
    label_counts = y.value_counts()

    # Plot the distribution using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(hue=label_counts.index,legend=False, y=label_counts.values, palette="viridis")
    plt.title("Distribution of Classes in `y`", fontsize=16)
    plt.xlabel("Classes (Alphabets)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#..................................................................



if __name__ == "__main__":
    csv_path = "C:\\Users\\HP\\Downloads\\archive\\A_Z Handwritten Data.csv"
    train_size = 50000  # Number of training examples
    test_size = 10000  # Number of testing examples

    # Load the data as a DataFrame from the CSV file
    data = load_data(csv_path)

    # Split the data into features and labels
    X, y = split_data(data)

    # Display number of unique classes and class distribution
    show_classes_info(y)

    # Display sample images
    show_images(X, y, 10)

    # Normalize pixel values to [0, 1] range by dividing by 255 (max pixel value)
    X_normalized =  X.to_numpy(dtype="float32") / 255.0
    print(f"\nX_normalized shape: {X_normalized.shape}")

    #...............................................................




