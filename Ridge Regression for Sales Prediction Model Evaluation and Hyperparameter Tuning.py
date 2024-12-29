# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv('WA_Marketing-Campaign.csv')

# One-hot encoding for categorical columns 
df = pd.get_dummies(df, columns=['MarketSize', 'LocationID'], drop_first=True)

# Define features (X) and target variable (y)
X = df.drop('SalesInThousands', axis=1)
y = df['SalesInThousands']

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a pipeline with data scaling and Ridge regression model
pipeline = make_pipeline(
    StandardScaler(),          # Scaling the data to standardize it
    Ridge(alpha=1.0)           # Ridge regression (L2 regularization)
)

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Model evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared score

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Get the coefficients from the trained Ridge model
model = pipeline.named_steps['ridge']
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Actual vs Predicted Sales 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.title("Actual vs Predicted Sales", fontsize=14)
plt.show()

# Regression Line (Red line indicates the linear regression fit)
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.title("Regression Line for Actual vs Predicted Sales", fontsize=14)
plt.show()

# Residuals Histogram (Distribution of the residuals/errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Residuals Distribution", fontsize=14)
plt.show()

# Hyperparameter tuning using GridSearchCV for Ridge Regression
ridge = Ridge()
param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters from Grid Search:", grid_search.best_params_)

# Polynomial feature transformation (degree 2) to capture non-linear relationships
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Standard scaling on the polynomial features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Cross-validation with Ridge regression using 5-fold cross-validation
scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {scores}")
print(f"Mean Cross-validation MSE: {np.mean(scores)}")

# Fit Ridge regression on polynomial features
ridge_poly = Ridge(alpha=0.1)
ridge_poly.fit(X_poly, y)

# Evaluate model performance using R-squared score on original data
print(f'R^2 score for polynomial features: {ridge_poly.score(X_poly, y)}')

# Split polynomial features into train and test sets for final evaluation
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
ridge_poly.fit(X_train_poly, y_train)

# Evaluate model performance on the test set
print(f'R^2 score on test set for polynomial features: {ridge_poly.score(X_test_poly, y_test)}')
