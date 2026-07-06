import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. DATA PREPARATION
# ==========================================
# 1 = Bought Pastry, 0 = Did Not Buy Pastry
data = {
    "Age": [22, 45, 30, 50, 25, 35, 40, 28, 55, 32],
    "Cups_Per_Week": [2, 7, 4, 8, 3, 5, 6, 2, 9, 4],
    "Distance_Miles": [5.0, 1.2, 3.5, 0.5, 6.2, 2.0, 1.5, 7.0, 0.8, 4.0],
    "Bought_Pastry": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0] 
}

df = pd.DataFrame(data)

X = df.drop("Bought_Pastry", axis=1)
y = df["Bought_Pastry"]

# ==========================================
# 2. DATA SPLITTING & SCALING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pro-Tweak: Scale the features so Age and Distance are on the same mathematical playing field
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ==========================================
# 4. PREDICTION & EVALUATION
# ==========================================
predictions = model.predict(X_test_scaled)
score = accuracy_score(y_test, predictions)

print("--- AI Barista Model Results ---")
print(f"Predictions: {predictions}")
print(f"Real Answers: {y_test.values}")
print(f"Model Accuracy: {score * 100:.2f}%")