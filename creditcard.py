import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "transactions_100.xlsx"
df = pd.read_excel(file_path)

# Convert 'time' into numeric feature (extracting the hour)
df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))

# Convert categorical variables into numerical ones
df = pd.get_dummies(df, columns=['location', 'merchant_category'])

# Splitting data into features and target variable
X = df.drop(columns=['transaction_id', 'time', 'is_fraud'])
y = df['is_fraud']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the fraud detection model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the predictions to a new Excel file
df_test = X_test.copy()
df_test['actual_fraud'] = y_test
df_test['predicted_fraud'] = y_pred
df_test.to_excel("fraud_detection_results.xlsx", index=False)

print("Fraud detection analysis saved to fraud_detection_results.xlsx")

# Visualization: Fraud Frequency by Hour
plt.figure(figsize=(10,6))
sns.histplot(df[df['is_fraud'] == 1]['hour'], bins=24, kde=True, color='red', label="Fraud Transactions")
sns.histplot(df[df['is_fraud'] == 0]['hour'], bins=24, kde=True, color='blue', label="Non-Fraud Transactions")
plt.xlabel("Transaction Hour")
plt.ylabel("Frequency")
plt.title("Fraud vs Non-Fraud Transactions by Hour")
plt.legend()
plt.savefig("fraud_hour_distribution.png")
plt.show()
