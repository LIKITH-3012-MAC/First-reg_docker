import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1], [2], [5]])   # study hours
y = np.array([12, 35, 100])     # marks

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved as model.pkl")

