# Diabetes Health Indicators Dataset

This project is based on the Diabetes Health Indicators Dataset available on Kaggle. The aim of this project is to create predictive models for diabetes risk using machine learning techniques. In this project, we used the "diabetes_binary_health_indicators_BRFSS2015.csv" file, which is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015.

## Dataset Description

The "diabetes_binary_health_indicators_BRFSS2015.csv" file is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable is Diabetes_binary, which has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced.

## Repository Structure

The repository has the following structure:

- archive: moved to archive
- data: contains the diabetes_binary_health_indicators_BRFSS2015.csv file
- helpers: contains the methods.py file, which contains helper functions used in the notebooks
- Clustering (1).ipynb: contains the clustering analysis performed on the dataset
- Clustering2.ipynb: contains the modified version of the clustering analysis
- EDA - Logistic Regression coefficients.ipynb: contains the analysis of the logistic regression coefficients
- Explanatory_Data_Analysis.ipynb: contains the exploratory data analysis performed on the dataset
- FactorAnalysisOfMixedData.ipynb: contains the factor analysis of mixed data performed on the dataset
- Feature_Importance.ipynb: contains the analysis of the feature importance in the dataset
- Final_Modeling.ipynb: contains the final version of the modeling
- LightGBM_final_model.sav: contains the final version of the LightGBM model
- LightGBM_model.sav: contains the tuned version of the LightGBM model
- Modeling.ipynb: contains the modeling analysis
- Outliers (1).ipynb: contains the analysis of the outliers in the dataset
- PCA.ipynb: contains the analysis of the principal component analysis of the dataset
- PCA_withAddedGraph (1).ipynb: contains the analysis of the principal component analysis with an added graph
- README.md: this file
- methods.py: contains the helper functions used in the notebooks
- requirements.txt: contains the required packages for running the notebooks
- t-sne.ipynb: contains the analysis of the t-SNE algorithm on the dataset
- xgb_feature_importances.pickle: contains the analysis of the feature importance using the XGBoost algorithm

## How to Run the Notebooks

To run the notebooks, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required packages by running the command "pip install -r requirements.txt".
3. Navigate to the directory where the notebooks are located.
4. Open the notebook in Jupyter Notebook.
5. Run the notebook cells to reproduce the results.

Note: Some of the notebooks may take a long time to run due to the size of the dataset and the complexity of the analysis.

## Conclusion
In this project, we analyzed the Diabetes Health Indicators Dataset using various machine learning techniques. We created predictive models for diabetes risk and analyzed the feature importance in the dataset. The results of this analysis can be used to identify individuals at risk for diabetes and to develop targeted interventions to prevent or manage the disease.

In this project, we analyzed the Diabetes Health Indicators Dataset using various machine learning techniques. We created predictive models for diabetes risk and analyzed the feature importance in the dataset. The results of this analysis can be used to identify individuals at risk for diabetes and to develop targeted interventions to prevent or manage the disease.
