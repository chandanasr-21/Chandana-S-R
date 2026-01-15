"""
BMI Classification Project
Predict BMI category based on BMI value.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
# BMI values vs category
# 0 = Underweight, 1 = Normal, 2 = Overweight, 3 = Obese
X = [
    [16], [17], [18],
    [19], [20], [21], [22],
    [24], [25], [26], [27],
    [30], [32], [35]
]

y = [
    0, 0, 0,      # Underweight
    1, 1, 1, 1,   # Normal
    2, 2, 2, 2,   # Overweight
    3, 3, 3       # Obese
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("BMI Classification Accuracy:", accuracy)

# Predict BMI category
bmi_value = [[26]]
prediction = model.predict(bmi_value)

categories = ["Underweight", "Normal", "Overweight", "Obese"]
print("Predicted BMI Category:", categories[prediction[0]])
