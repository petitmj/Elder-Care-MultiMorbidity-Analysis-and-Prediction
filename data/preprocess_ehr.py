import pandas as pd

# List of chronic diseases to extract
chronic_diseases = [
    "hypertension", "diabetes", "heart_disease", "chronic_respiratory",
    "arthritis", "dementia", "stroke", "osteoporosis", "cancer"
]

# Load EHR data
input_path = "data/EHR.csv"
df = pd.read_csv(input_path)

# Preprocess: create binary columns for each disease
for disease in chronic_diseases:
    df[disease] = df["apacheadmissiondx"].str.contains(disease, case=False, na=False).astype(int)

# Save cleaned data
output_path = "data/EHR_Cleaned.csv"
df.to_csv(output_path, index=False)

print(f"Cleaned EHR data saved to {output_path}")
