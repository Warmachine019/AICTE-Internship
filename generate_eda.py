# Code to generate the EDA Visualisation Diagram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

excel_path = "Data_Set.xlsx"

excel_file = pd.ExcelFile(excel_path)

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

dataframes = []
for sheet in commodity_sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    # Keep only the needed columns
    if all(col in df.columns for col in feature_cols + [target_col]):
        df = df[feature_cols + [target_col]].dropna()
        dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

# EDA correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(combined_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of All Combined Sheets")
plt.tight_layout()

plt.savefig("eda_correlation_heatmap.png", dpi=300)

plt.show()
