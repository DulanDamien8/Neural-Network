import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Experiment with different hyperparameters for the neural network
hidden_layer_sizes = [(50,), (100,), (50, 50), (100, 50)]
activation_functions = ['relu', 'tanh']
learning_rates = ['constant', 'adaptive']

results = []

for hidden_layers in hidden_layer_sizes:
    for activation in activation_functions:
        for learning_rate in learning_rates:
            # Create and train the neural network
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, learning_rate=learning_rate,
                                max_iter=500, random_state=42)
            mlp.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = mlp.predict(X_test)
            accuracy = mlp.score(X_test, y_test)
            results.append((hidden_layers, activation, learning_rate, accuracy))

# Display the results of different parameter sets
print("Summary")
for result in results:
    print(f"hidden_layers={result[0]}, activation={result[1]}, learning_rate={result[2]}, accuracy={result[3]}")

# Display first 10 images from the dataset
fig, ax = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[i // 5, i % 5].imshow(faces.images[i + 200], cmap='bone')

plt.show()
