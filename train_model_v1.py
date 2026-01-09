import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Data
url = "https://raw.githubusercontent.com/uzairizzuddin/Penguin-model-monitoring-dashboard/refs/heads/main/penguins.csv"
df = pd.read_csv(url)

# 2. Remove rows with missing values
df = df.dropna()

# 3. Features & Target
# Bill Length, Bill Depth, Flipper Length, Body Mass
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# 4. Train Model v1 (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 5. Save Model
joblib.dump(model, 'model_v1.pkl')
print("Model v1 saved successfully!")