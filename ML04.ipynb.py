# 4] Implement and visualize k-Nearest Neighbors classification.
#    Tune k and compare accuracy.

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("Social_Network_Ads.csv")

print(df.head())

# ------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------
print(df.shape)  # (rows, columns)

print(df['Purchased'].value_counts())

# Scatter Plot to show non-linear data
plt.scatter(df['Age'], df['EstimatedSalary'], cmap='coolwarm', c=df['Purchased'])
plt.show()

# ------------------------------------------------------------
# 3. Feature Selection
# ------------------------------------------------------------
# Select features and target
X = df[['Age', 'EstimatedSalary']]  # Features
y = df['Purchased']                 # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()  # mean=0, std=1

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# 4. Model Building
# ------------------------------------------------------------
# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)

# Classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize KNN classification
plt.figure(figsize=(8, 6))

plt.scatter(X_test[:, 0], X_test[:, 1],
            c=y_pred, cmap='viridis', s=60)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("KNN Classification")
plt.show()

# Tune K
# Train no of k values
k_values = range(1, 16)
accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

    print("K =", k, "Accuracy =", round(accuracy, 3))

# ------------------------------------------------------------
# 5. Model Evaluation
# ------------------------------------------------------------
# Plot K vs Accuracy
plt.figure(figsize=(8, 5))

plt.plot(k_values, accuracies, marker='o')  # Line Plot

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN: K Value vs Accuracy")

plt.xticks(k_values)
plt.grid()

plt.show()

# Finding best k and accuracy value
best_k = k_values[accuracies.index(max(accuracies))]

print("Best K:", best_k)
print("Best Accuracy:", max(accuracies))
