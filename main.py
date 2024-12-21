import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


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
    plt.figure(figsize=(5, 5))
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
    
    
    
def train_and_evaluate_linear_SVM(X_train, y_train, X_test, y_test):
    print("\nTraining Linear SVM using SVC...")
    linear_svm = SVC(kernel="linear", random_state=42)
    linear_svm.fit(X_train, y_train)
    y_pred_linear = linear_svm.predict(X_test)

    print("\nLinear SVM Evaluation:")
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Accuracy: {accuracy_linear:.4f}")
    print(classification_report(y_test, y_pred_linear))
    sns.heatmap(confusion_matrix(y_test, y_pred_linear), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: Linear SVM")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    print(f"\nAverage F1 Score: {f1_score(y_test, y_pred_linear, average='weighted'):.4f}")
    
    # Train and evaluate Non-Linear SVM (RBF kernel)
def train_and_evaluate_rbf_SVM(X_train, y_train, X_test, y_test):
    print("\nTraining Non-Linear SVM using RBF Kernel...")
    rbf_svm = SVC(kernel="rbf", random_state=42)
    rbf_svm.fit(X_train, y_train)
    y_pred_rbf = rbf_svm.predict(X_test)

    print("\nNon-Linear SVM (RBF Kernel) Evaluation:")
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"Accuracy: {accuracy_rbf:.4f}")
    print(classification_report(y_test, y_pred_rbf))
    sns.heatmap(confusion_matrix(y_test, y_pred_rbf), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: Non-Linear SVM (RBF Kernel)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    print(f"\nAverage F1 Score: {f1_score(y_test, y_pred_rbf, average='weighted'):.4f}")



if __name__ == "__main__":
    # File path
    csv_path = r"C:\Users\10\Documents\ML Project\archive\A_Z Handwritten Data.csv"

    # Load data
    data = load_data(csv_path)
    if data is None:
        exit("Failed to load data. Exiting...")

    # Shuffle data (to ensure balanced sampling)
    data = shuffle(data, random_state=42)

    # Split features and labels
    X, y = split_data(data)
    if X is None or y is None:
        exit("Failed to split data. Exiting...")

    # Show class information
    show_classes_info(y)

    # Show sample images
    show_images(X, y, 10)

    # Normalize pixel values
    X_normalized = X / 255.0
    X_normalized = X_normalized.astype('float32')  # Convert to float32 for memory efficiency
    print(f"\nX_normalized shape: {X_normalized.shape}")

    # Apply PCA to reduce dimensionality
    print("\nApplying PCA...")
    pca = PCA(0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_normalized)
    print(f"X_pca shape: {X_pca.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")

    # Split data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42
    )

    print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    train_and_evaluate_rbf_SVM(X_train, y_train, X_test, y_test)
