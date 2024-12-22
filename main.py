
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


def load_data(csv_path_: str):
    # Load dataset
    data = pd.read_csv(csv_path_)
    print(data.head())

    return data


def split_data(data: pd.DataFrame):
    # split data to features and label
    X = data.drop('0', axis=1)
    y = data['0']

    return X, y


def preprocess_labels(y: pd.Series):
    y_encodingHot = to_categorical(y, num_classes=26)  # One-hot encode
    return y_encodingHot


def show_images(X: pd.DataFrame, y: pd.Series, n_images: int):
    # Display sample images

    # Reshape into 28x28
    X_images = X.values.reshape(-1, 28, 28)

    # Plot the first 10 images
    plt.figure(figsize=(10, 5))
    for i in range(n_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_images[i], cmap="gray")
        plt.title(f"Label: {y[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_classes_info(y: pd.Series):
    # Show number of unique classes
    print(f"\nNumber of unique classes: {y.nunique()}")

    # Show class distribution
    label_counts = y.value_counts()

    # Plot the distribution using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Distribution of Classes in `y`", fontsize=16)
    plt.xlabel("Classes (Alphabets)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# ........................................

# Neural Network
def split_dataset(X, y, test_size=0.3, val_size=0.5):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# build Model 1
def build_neuralnetwork_model_1():
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation='relu'),  # 128 neurons
        Dense(64, activation='relu'),  # 64 neurons
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Build Model 2
def build_neuralnetwork_model_2():
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(256, activation='relu'),  # First layer>  256 neurons
        Dense(128, activation='tanh'),  # second layer>  128 neurons
        Dense(64, activation='relu'),  # third layer>  64 neurons
        Dense(32, activation='relu'),  # 4 layer> 32 neurons
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_plotNN(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    # Train the model
    trained_model = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    plt.figure(figsize=(12, 5))

    # accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(trained_model.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(trained_model.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # loss plot
    plt.subplot(1, 2, 2)
    plt.plot(trained_model.history['loss'], label='Training Loss', color='blue')
    plt.plot(trained_model.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return trained_model


def evaluate_NN(model, x_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred = model.predict(x_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_actual_classes = y_test.argmax(axis=1)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_actual_classes, y_pred_classes)
    letter = []
    for i in range(1, 27):
        letter.append(chr(96 + i))
    sns.heatmap(conf_matrix, annot=True, cmap='binary', fmt='.0f', xticklabels=letter, yticklabels=letter)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.show()

    # F1 Score
    f1 = f1_score(y_actual_classes, y_pred_classes, average='macro')
    print(f"\nAverage F1-Score: {f1:.2f}")


# ...............................................
def create_image_for_char(char, data, labels):
    label = ord(char.upper()) - 65
    char_indices = np.where(labels == label)[0]
    index = char_indices[0]
    image = data.iloc[index].values.reshape(28, 28).astype('float32')
    image = image / 255.0  # Normalize
    return image


def test_NN_with_name(name, model, data, labels):
    predictions = []
    for letter in name:
        try:
            image = create_image_for_char(letter, data, labels)
            final_image = image.reshape(1, 28, 28)
            pred = model.predict(final_image)
            pred_letter = np.argmax(pred)
            predictions.append(pred_letter)
        except ValueError as e:
            print(e)
            predictions.append(None)

    pred_letters = []


    for pred in predictions:
     if pred is not None:
        pred_letters.append(chr(pred + 65))

     print(f"Predicted letters: {''.join(pred_letters)}")

     plt.figure(figsize=(10, 1))
    for i, letter in enumerate(name):
       plt.subplot(1, len(name), i + 1)
    try:
        image = create_image_for_char(letter, data, labels)
        plt.imshow(image, cmap='gray')
        plt.title(f"Pred: {pred_letters[i]}")
    except ValueError:
        plt.title("NA")
    plt.axis('off')
plt.show()

# .......................................
#SVM
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
#......................................................................
#logistic
class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.classes = None
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Find all unique classes
        n_classes = len(self.classes)

        # Initialize weights and biases for each class
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        # Train one model per class (One-vs-All)
        for idx, c in enumerate(self.classes):
            y_binary = (y == c).astype(int)  # Convert labels to binary for this class

            for _ in range(self.n_iters):
                linear_model = np.dot(X, self.weights[idx]) + self.bias[idx]
                y_predicted = self._sigmoid(linear_model)

                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary))
                db = (1 / n_samples) * np.sum(y_predicted - y_binary)

                # Update parameters
                self.weights[idx] -= self.lr * dw
                self.bias[idx] -= self.lr * db

                # Calculate training loss and accuracy
                loss = -np.mean(y_binary * np.log(y_predicted + 1e-9) + (1 - y_binary) * np.log(1 - y_predicted + 1e-9))
                accuracy = np.mean((y_predicted >= 0.5) == y_binary)

                if len(self.training_loss) < self.n_iters:
                    self.training_loss.append(loss)
                    self.training_accuracy.append(accuracy)

                # Validation loss and accuracy
                if X_val is not None and y_val is not None:
                    y_val_binary = (y_val == c).astype(int)
                    val_linear_model = np.dot(X_val, self.weights[idx]) + self.bias[idx]
                    y_val_predicted = self._sigmoid(val_linear_model)
                    val_loss = -np.mean(
                        y_val_binary * np.log(y_val_predicted + 1e-9) +
                        (1 - y_val_binary) * np.log(1 - y_val_predicted + 1e-9)
                    )
                    val_accuracy = np.mean((y_val_predicted >= 0.5) == y_val_binary)

                    if len(self.validation_loss) < self.n_iters:
                        self.validation_loss.append(val_loss)
                        self.validation_accuracy.append(val_accuracy)

    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias
        probabilities = self._sigmoid(linear_model)
        return self.classes[np.argmax(probabilities, axis=1)]  # Pick class with highest probability

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        avg_f1 = f1_score(y_test, y_pred, average='macro')
        acc = np.mean(y_pred == y_test)
        return cm, avg_f1, acc

    def plot_curves(self):
        plt.figure(figsize=(12, 5))

        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(self.training_loss, label='Training Loss')
        if self.validation_loss:
            plt.plot(self.validation_loss, label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(self.training_accuracy, label='Training Accuracy')
        if self.validation_accuracy:
            plt.plot(self.validation_accuracy, label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

#...........................................................
if __name__ == "__main__":
    csv_path = "../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv"
    data = load_data(csv_path)
    X, y = split_data(data)

    # Show Class Distribution and Sample Images
    show_classes_info(y)

    show_images(X, y, 10)

    # Normalize Data
    X_normalized = X.to_numpy(dtype="float32") / 255.0
    print(f"\nX_normalized shape: {X_normalized.shape}")

    # .....................................

    y_encoded = preprocess_labels(y)

    X_normalized_reshaped = X_normalized.reshape(-1, 28, 28)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_normalized_reshaped, y_encoded)

    # Train and Evaluate Model 1
    print("\nTraining Model 1:")
    model1 = build_neuralnetwork_model_1()
    training1 = train_and_plotNN(model1, X_train, y_train, X_val, y_val)
    print("\nEvaluating Model 1:")
    evaluate_NN(model1, X_test, y_test)

    # ...................................

    # Train and Evaluate Model 2
    print("\nTraining Model 2:")
    model2 = build_neuralnetwork_model_2()
    training2 = train_and_plotNN(model2, X_train, y_train, X_val, y_val)
    print("\nEvaluating Model 2:")
    evaluate_NN(model2, X_test, y_test)
    # ...............................................

    # save best model
    if training2.history['val_accuracy'][-1] > training1.history['val_accuracy'][-1]:
        model2.save('best_nn_model.h5')
        print("Model 2 is the best model.")
        best_model = model2
    else:
        model1.save('best_nn_model.h5')
        print("Model 1 is the best model.")
        best_model = model1
    # ...........................................
    # reload best model
    best_model = keras.models.load_model('best_nn_model.h5')
    evaluate_NN(best_model, X_test, y_test)
    name = "Shahd"
    name1 = "Aliaa"
    name2 = "Nahla"
    test_NN_with_name(name, best_model, X, y)
    test_NN_with_name(name1, best_model, X, y)
    test_NN_with_name(name2, best_model, X, y)
    #......................................................

    # Apply PCA to reduce dimensionality
    print("\nApplying PCA...")
    pca = PCA(0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_normalized)
    print(f"X_pca shape: {X_pca.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")

    # Split data into training and testing sets (70% training, 30% testing)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X_pca, y, test_size=0.3, random_state=42
    )

    print(f"\nX_train shape: {X_train1.shape}, y_train shape: {y_train1.shape}")
    print(f"X_test shape: {X_test1.shape}, y_test shape: {y_test1.shape}")
    train_and_evaluate_rbf_SVM(X_train1, y_train1, X_test1, y_test1)

    #.................................................................................

    test_sample_size = 50000  # Set the number of test samples to use for evaluation
    train_sample_size = 50000  # Set the number of training samples to use for training

    # Split the data into training, validation, and testing datasets
    X_train2, X_temp2, y_train2, y_temp2 = train_test_split(X_normalized, y, test_size=0.4, random_state=42)
    X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp2, y_temp2, test_size=0.5, random_state=42)

    # Use only a subset of the training data for training
    X_train_subset = X_train2[:train_sample_size]
    y_train_subset = y_train2[:train_sample_size]

    # Use only a subset of the test data for evaluation
    X_test_subset = X_test2[:test_sample_size]
    y_test_subset = y_test2[:test_sample_size]

    # Initialize and train the model
    model = LogisticRegression(learning_rate=0.05, n_iters=2000)
    model.fit(X_train_subset, y_train_subset, X_val2, y_val2)  # Train on the subset of the training data

    # Plot training and validation curves
    model.plot_curves()

    # Evaluate the model on the subset of the test set
    confusion_matrix, avg_f1_score, accuracy = model.evaluate(X_test_subset, y_test_subset)
    print("Confusion Matrix:\n", confusion_matrix)
    print("Average F1 Score:", avg_f1_score)
    print("Accuracy:", accuracy)









