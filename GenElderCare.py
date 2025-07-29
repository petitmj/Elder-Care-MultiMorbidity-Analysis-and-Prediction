import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from pyvis.network import Network
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import gudhi  # For TDA (Topological Data Analysis)

# Predefined chronic diseases
chronic_diseases = [
    "hypertension", "diabetes", "heart_disease", "chronic_respiratory",
    "arthritis", "dementia", "stroke", "osteoporosis", "cancer"
]

# Streamlit App Title
st.title("Elder Care MultiMorbidity Analysis and Prediction")

# Sidebar for Data Uploads
st.sidebar.header("Upload Your Data")
st.sidebar.write("Please upload at least one dataset in CSV format:")

# File upload widgets
ehr_file = st.sidebar.file_uploader("Upload EHR Data (Electronic Health Records)", type="csv")
sdo_file = st.sidebar.file_uploader("Upload SDOH Data (Social Determinants of Health)", type="csv")
wearable_file = st.sidebar.file_uploader("Upload Wearable Data", type="csv")
multiomics_file = st.sidebar.file_uploader("Upload Multi-Omics Data (Genomics, Transcriptomics, etc.)", type="csv")

# Function to optimize memory usage
def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

# Function to standardize the patient_id column across datasets
def standardize_patient_id_column(df):
    for col in df.columns:
        if "id" in col.lower():
            df.rename(columns={col: "patient_id"}, inplace=True)
            df["patient_id"] = df["patient_id"].astype(str)
            return df
    raise ValueError("No column resembling 'patient_id' found in the dataset.")

# Load uploaded files into DataFrames (if available)
dataframes = []
if sdo_file:
    sdo_data = pd.read_csv(sdo_file)
    sdo_data = optimize_memory(sdo_data)
    sdo_data = standardize_patient_id_column(sdo_data)
    dataframes.append(sdo_data)
if ehr_file:
    ehr_data = pd.read_csv(ehr_file)
    ehr_data = optimize_memory(ehr_data)
    ehr_data = standardize_patient_id_column(ehr_data)
    dataframes.append(ehr_data)
if wearable_file:
    wearable_data = pd.read_csv(wearable_file)
    wearable_data = optimize_memory(wearable_data)
    wearable_data = standardize_patient_id_column(wearable_data)
    dataframes.append(wearable_data)
if multiomics_file:
    multiomics_data = pd.read_csv(multiomics_file)
    multiomics_data = optimize_memory(multiomics_data)
    multiomics_data = standardize_patient_id_column(multiomics_data)
    dataframes.append(multiomics_data)

# Proceed if at least one dataset is available
if dataframes:
    # Merge all uploaded datasets
    merged_data = dataframes[0]
    for df in dataframes[1:]:
        merged_data = pd.merge(merged_data, df, on="patient_id", how="inner")

    # Dynamically identify chronic disease columns in the uploaded data
    available_columns = set(merged_data.columns)
    chronic_diseases_in_data = [disease for disease in chronic_diseases if disease in available_columns]

    if not chronic_diseases_in_data:
        st.error("No matching chronic disease columns found in the dataset. Please verify the data.")
    else:
        st.write("### Chronic Diseases Found in Data")
        st.write(chronic_diseases_in_data)

        # Reduce merged dataset to required columns
        merged_data = merged_data[chronic_diseases_in_data + ['patient_id']]

        # Fill missing values with 0
        merged_data.fillna(0, inplace=True)

        # Disease-specific data extraction
        st.header("Extract Data for a Specific Disease")
        selected_disease = st.selectbox(
            "Select a Disease to Extract Data", options=chronic_diseases_in_data
        )
        disease_specific_data = merged_data[merged_data[selected_disease] > 0]
        st.write(f"#### Data for {selected_disease.title()}")
        st.dataframe(disease_specific_data)

        # Create Co-occurrence Matrix for Disease-Disease Interaction
        st.header("Disease-Disease Interaction Network")
        st.write("Visualizing the co-occurrence of chronic diseases.")

        # Sparse representation for large data
        disease_data = merged_data[chronic_diseases_in_data]
        disease_data_sparse = csr_matrix(disease_data.values)
        co_occurrence = disease_data_sparse.T.dot(disease_data_sparse).toarray()

        # Advanced Matrix Factorization: Apply PCA or NMF
        method_choice = st.selectbox("Choose Matrix Factorization Method", ["NMF", "PCA"])
        if method_choice == "NMF":
            nmf = NMF(n_components=3, init='random', random_state=42)
            W = nmf.fit_transform(co_occurrence)
            H = nmf.components_
            co_occurrence_reconstructed = np.dot(W, H)
        else:
            pca = PCA(n_components=9)
            co_occurrence_reconstructed = pca.fit_transform(co_occurrence)

        # Function to generate network graph with adjusted edge thickness based on reconstructed matrix
        def generate_network(selected_diseases):
            disease_colors = {
                "hypertension": "#1f77b4", "diabetes": "#ff7f0e", "heart_disease": "#2ca02c",
                "chronic_respiratory": "#d62728", "arthritis": "#9467bd", "dementia": "#8c564b",
                "stroke": "#e377c2", "osteoporosis": "#7f7f7f", "cancer": "#bcbd22"
            }

            # Create a Pyvis Network object
            disease_network = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
            disease_network.force_atlas_2based()  # Use Force Atlas layout for better clarity

            # Add selected diseases as nodes
            for disease in selected_diseases:
                disease_network.add_node(
                    disease, label=disease.replace("_", " ").title(), color=disease_colors.get(disease, "#000000")
                )

            # Add edges based on reconstructed co-occurrence matrix
            max_cooccurrence = co_occurrence_reconstructed.max()
            for i, disease_1 in enumerate(selected_diseases):
                for j, disease_2 in enumerate(selected_diseases):
                    if i < j:  # Avoid duplicating edges or self-loops
                        continue
                    weight = co_occurrence_reconstructed[i, j]
                    if weight > 0:  # Only include edges with non-zero weight
                        edge_thickness = (weight / max_cooccurrence) * 100  # Increase edge thickness
                        disease_network.add_edge(
                            disease_1, disease_2, value=edge_thickness, title=f"Co-occurrence: {int(weight)}"
                        )

            # Save the network as an HTML file
            html_file = "disease_network.html"
            disease_network.save_graph(html_file)

            # Render the interactive graph using Streamlit's HTML component
            try:
                with open(html_file, "r") as f:
                    st.components.v1.html(f.read(), height=800)
            except Exception as e:
                st.error(f"Error rendering network: {e}")

        # Disease selection for dynamic update
        selected_diseases = st.multiselect(
            "Select Diseases to Visualize", options=chronic_diseases_in_data, default=chronic_diseases_in_data
        )

        # Generate the network graph based on the selected diseases
        generate_network(selected_diseases)

        # Random Forest Classifier for Multi-Morbidity Prediction
        st.header("Multi-Morbidity Prediction")
        st.write("Predict the likelihood of disease co-occurrence using a Random Forest Classifier.")
        
        # Prepare data for classification (using co-occurrence as features)
        X = co_occurrence_reconstructed
        y = np.random.choice([0, 1], size=X.shape[0])  # Binary target variable (replace with actual target)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Prediction Accuracy: {accuracy * 100:.2f}%")

else:
    st.warning("Please upload at least one dataset to proceed.")
























       














