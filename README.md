# ExampleNotebooks
This is a repository containing some jupyter notebooks that I used to explore various concepts in classification and model training.

##  discrete user action sequence analysis_transition prob and ngrams.ipynb
This notebook explores discrete user action/transaciton sequences and grouping those sequences, in general, and for defining specific subsequences. The goal is to show methods to compare different types of subsequences of user activity, by measuring thier transition probablity differences between actions, and the ngrams of actions, as well as the replacement of common action sequences with meta_action sequences as to increase visability for transition prob and ngram analysis.

##  nonlinear data classifier methods.ipynb
A subset of nonlinear data classifier methods.ipynb is displayed on a webpage that displays the animation of the decision boundaries for Neural Networks of various sizes during training. 
You can find the webpage here:
Neural Network Model Performance Across Datasets 
https://nockbarry.github.io/ExampleNotebooks/

The rest of the notebook explores other classifcation methods on the same data, the results for which are viewable in the .ipynb notebook

## randomstring_classification.ipynb
This notebook explores an NLP problem of classifing short strings of gibberish letters. It contains a synthetic data generation procedure for plasuible real data from Faker and gibberish/random string emails. It tests a few encodings, count, ngram, tfidf , trandiitonal classification models using sklearn and keras neural network models, amd anomaly detection. Compares undersampling data and minimal data requirments for model performance using hold out data.
