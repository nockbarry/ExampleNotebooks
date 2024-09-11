# ExampleNotebooks
This is a repository containing some jupyter notebooks that I used to explore various concepts in classification and model training.

## Model Test Suite
The ModelTestSuite is designed to assist in the evaluation of different machine learning models, feature transformations, and train-test split strategies for fraud detection. Its core purpose is to simplify the process of comparing various models across different feature configurations and cross-validation techniques, including Leave-One-Category-Out (LOCO) cross-validation.

This tool provides an integrated framework to handle the entire fraud detection pipeline, from data preprocessing to model evaluation, while ensuring flexibility in how the data is split and the models are trained. The suite is particularly useful for users looking to test multiple combinations of feature sets and models, all while gaining insights into how various cross-validation methods, including time-based and LOCO CV, affect model performance.
Key Features:

- Flexible Train-Test Splits: The suite allows for various train-test splitting techniques, including random splits, time-based splits (ensuring no future data leakage), and LOCO CV. This makes it easy to evaluate models under different data segmentation strategies and understand the impact of each.

- Model Agnosticism: The framework supports multiple classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting, and XGBoost), allowing users to compare different algorithms easily within the same environment. All models can be trained and evaluated under the same conditions, ensuring consistency in comparison.

- Feature Transformations: It includes a wide range of feature preprocessing options, such as standard scaling, PCA for dimensionality reduction, and handling of collinear features. Additionally, the framework allows you to add unsupervised learning results, such as clustering labels, as new features, thus enriching the feature set.

- Leave-One-Category-Out CV: A key strength of the suite is its LOCO cross-validation feature. This method tests models by leaving out specific categories of data, making it especially useful for detecting how models perform on unseen categories or under specific group exclusions.

- End-to-End Fraud Detection Pipeline: The suite automates the full pipeline—from data preprocessing (handling missing values, scaling, and encoding) to model evaluation (with metrics like accuracy, precision, recall, F1-score, and ROC-AUC). It also supports SHAP values for model interpretability, clustering, anomaly detection, and data distribution drift analysis.

The primary goal of the ModelTestSuite is to facilitate the exploration and comparison of multiple combinations of feature configurations, models, and cross-validation techniques. By automating much of the repetitive setup work involved in testing these configurations, the tool helps users focus on interpreting results and improving model performance, rather than spending time on manual configuration.

## Bokeh Network Plots
Explores some interactions for a graph with different node types with edges that have timestamps in networkx plotting with bokeh.
Preview: https://html-preview.github.io/?url=https://github.com/nockbarry/ExampleNotebooks/blob/main/bokeh_network_plotting.html

##  discrete user action sequence analysis_transition prob and ngrams.ipynb
This notebook explores discrete user action/transaciton sequences and grouping those sequences, in general, and for defining specific subsequences. The goal is to show methods to compare different types of subsequences of user activity, by measuring thier transition probablity differences between actions, and the ngrams of actions, as well as the replacement of common action sequences with meta_action sequences as to increase visability for transition prob and ngram analysis.

##  nonlinear data classifier methods.ipynb
A subset of nonlinear data classifier methods.ipynb is displayed on a webpage that displays the animation of the decision boundaries for Neural Networks of various sizes during training. 
You can find the webpage here:
Neural Network Model Performance Across Datasets 
https://nockbarry.github.io/ExampleNotebooks/

The rest of the notebook explores other classifcation methods on the same data, the results for which are viewable in the .ipynb notebook

## randomstring_classification.ipynb
This notebook explores an NLP problem of classifying short strings of gibberish letters. It contains a synthetic data generation procedure for plausible real data from Faker and gibberish/random string emails. It tests a few encodings, count, ngram, tfidf , trandiitonal classification models using sklearn and keras neural network models, as well as some anomaly detection methods. Compares model performance in prediction and compute time across all of these parameters.
<br>
The second document with the _undersampling name compares undersampling data and minimal data requirements for model performance using hold out data. 

## xgboost_model_analysis.ipynb

This Python notebook includes a collection of functions and library imports for machine learning applications, focusing on data preprocessing, model training, evaluation, and visualization. Libraries such as Numpy, Pandas, XGBoost, Matplotlib, Seaborn, SHAP, and Scikit-Learn are imported for various tasks. Functions are provided to preprocess data, split datasets, compute permutation importance, plot learning curves, tune hyperparameters, display search results, evaluate models, plot Partial Dependence Plots (PDPs), detect high leverage points, and analyze model residuals. This script is structured to support machine learning projects, particularly those involving classification tasks and model interpretability analyses.
