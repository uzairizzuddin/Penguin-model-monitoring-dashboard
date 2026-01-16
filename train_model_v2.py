import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Data
url = "https://raw.githubusercontent.com/uzairizzuddin/Penguin-model-monitoring-dashboard/refs/heads/main/penguins.csv"
df = pd.read_csv(url)
df = df.dropna()

# 2. Features & Target
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# 3. Train Model v2 (Random Forest)
# "Improved" model iteration.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save Model
joblib.dump(model, 'model_v2.pkl')
print("Model v2 saved successfully!")