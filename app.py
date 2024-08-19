import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
data = pd.read_csv("trainingData.csv")

# Split data into features and target
X = data.drop(["number_of_product_units", "generic_holiday"], axis=1)
y = data["number_of_product_units"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101)

# Scale the features
transformer = RobustScaler().fit(X_train)
X_train_scaled = transformer.transform(X_train)
X_val_scaled = transformer.transform(X_val)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=13, random_state=101)
model.fit(X_train_scaled, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    X = [to_predict_list]
    X_scaled = transformer.transform(X)
    return model.predict(X_scaled)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_from_page = request.form.to_dict()
        
        # Convert data to list of values
        to_predict_list = [int(data_from_page.get(key)) for key in ['product_type', 'cost_per_unit', 'time_delivery', 'revenue', 'day_of_week']]
        
        # Predict the specific instance
        result = ValuePredictor(to_predict_list)
        
        # Sample a random subset of validation data to compute MAE
        idx = np.random.choice(np.arange(len(X_val_scaled)), size=int(len(X_val_scaled) * 0.1), replace=False)
        sample_y_val = y_val.iloc[idx]
        sample_y_pred = model.predict(X_val_scaled[idx])
        
        mae = mean_absolute_error(sample_y_val, sample_y_pred)
        
        # Convert result to integer for display
        prediction_text = int(result[0])
        
        return render_template('index.html', prediction_text=prediction_text, mae=mae)

if __name__ == "__main__":
    app.run(debug=True)

