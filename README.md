# Elder Care MultiMorbidity Analysis and Prediction
This project provides an interactive machine learning-based solution for analyzing **multimorbidity**, the co-occurrence of multiple chronic diseases. By integrating data from various sources (EHR, SDOH, wearable, and multi-omics datasets), it uncovers disease relationships, builds interactive disease-disease networks, and predicts multimorbidity risks.

## Features

- **Data Integration**: Combines EHR, SDOH, wearable data, and multi-omics datasets for comprehensive analysis.
- **Interactive Disease Network**: Visualizes disease co-occurrence using a dynamically generated graph.
- **Dimensionality Reduction**: Offers matrix factorization (NMF) and PCA for uncovering latent patterns in multimorbidity.
- **Risk Prediction**: Uses machine learning (Random Forest) to predict the likelihood of multimorbidity.
- **Streamlit Interface**: Provides an easy-to-use interface for uploading datasets and interacting with the visualizations.

---

## Requirements

### **Dependencies**
The following Python libraries are required:
- `streamlit` (web application framework)
- `pandas` (data handling)
- `numpy` (numerical computing)
- `scipy` (for sparse matrix operations)
- `scikit-learn` (machine learning)
- `pyvis` (interactive network visualization)

Install all dependencies using:
```bash
pip install streamlit pandas numpy scipy scikit-learn pyvis
```

---

## Usage

### **1. Run the Application**
Start the Streamlit application with the following command:
```bash
streamlit run GenElderCare.py
```

### **2. Upload Datasets**
Upload your datasets in CSV format:
- **EHR Data**: Contains chronic disease records.
- **SDOH Data**: Includes demographic, social, and economic factors.
- **Wearable Data**: Captures physiological metrics (e.g., heart rate, steps).
- **Multi-Omics Data**: Includes genomics, transcriptomics, and proteomics.

### **3. Disease Network Visualization**
- Select diseases to visualize using a multiselect dropdown.
- View an interactive graph with:
  - **Nodes**: Representing diseases.
  - **Edges**: Representing co-occurrence strength (thickness based on relationship).

### **4. Multimorbidity Prediction**
- Train a **Random Forest classifier** using the reconstructed co-occurrence matrix.
- View prediction accuracy for multimorbidity risks.

---

## Project Structure

```plaintext
.
├── GenElderCare.py      # Main application script
├── requirements.txt     # List of dependencies
├── disease_network.html # Temporary file for disease graph visualization
├── data                 # synthetic data to test the model with
  ├── ehr_data.csv
  ├── multiomics_data.csv
  ├── sdo_data.csv
  ├── wearable_data.csv

```

---

## Algorithms Used

1. **Dimensionality Reduction**:
   - **NMF**: Non-negative matrix factorization uncovers latent disease groupings.
   - **PCA**: Principal Component Analysis reduces data dimensions for visualization.

2. **Machine Learning**:
   - **Random Forest Classifier**: Predicts multimorbidity risks based on disease co-occurrence patterns.

3. **Interactive Visualization**:
   - **Pyvis**: Generates a dynamic graph where diseases are nodes, and co-occurrence relationships are edges.

---

## Example Workflow

1. **Upload Datasets**:
   Upload your EHR, SDOH, wearable, or omics datasets through the sidebar in the Streamlit app.

2. **Explore Disease Interactions**:
   Select diseases of interest and generate an interactive disease-disease co-occurrence network.

3. **Predict Multimorbidity**:
   Train the Random Forest model and view the accuracy score for risk prediction.

---

## Potential Impact

This project enables:
- Early identification of multimorbidity risks.
- Better understanding of disease relationships.
- Insights into personalized treatments and public health interventions.

---

## Future Enhancements

- Add support for **longitudinal datasets** to capture temporal trends in multimorbidity.
- Include additional machine learning models for advanced predictions.
- Incorporate explainable AI (XAI) to provide clinical interpretability.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

For questions or contributions, feel free to open an issue or submit a pull request on this GitHub repository!
