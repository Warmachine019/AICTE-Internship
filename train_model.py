import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# File and Sheet handling
excel_path = "Data_Set.xlsx"
excel_file = pd.ExcelFile(excel_path)

# Select only Summary_Commodity sheets
commodity_sheets = [sheet for sheet in excel_file.sheet_names if sheet.endswith('_Summary_Commodity')]

feature_cols = [
    'Supply Chain Emission Factors without Margins',
    'Margins of Supply Chain Emission Factors',
    'DQ ReliabilityScore of Factors without Margins',
    'DQ TemporalCorrelation of Factors without Margins',
    'DQ GeographicalCorrelation of Factors without Margins',
    'DQ TechnologicalCorrelation of Factors without Margins',
    'DQ DataCollection of Factors without Margins'
]
target_col = 'Supply Chain Emission Factors with Margins'

# Loading and combining all the sheets
dataframes = []
for sheet in commodity_sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if all(col in df.columns for col in feature_cols + [target_col]):
        df = df[feature_cols + [target_col]].dropna()
        dataframes.append(df)

# Combine data from all sheets
combined_df = pd.concat(dataframes, ignore_index=True)

# Splitting the data
X = combined_df[feature_cols]
y = combined_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the RFR Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Testing the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Trained on {len(X)} rows from {len(commodity_sheets)} sheets")
print(f"📉 Mean Squared Error: {mse:.6f}")

# Saving the Random forest mode, scaler and feature columns
joblib.dump(model, '../Final Project/models/random_forest_model.pkl')
joblib.dump(scaler, '../Final Project/models/scaler.pkl')
joblib.dump(feature_cols, '../Final Project/models/feature_columns.pkl')  # Save feature order for inference
print("💾 Saved: random_forest_model.pkl, scaler.pkl, feature_columns.pkl")
