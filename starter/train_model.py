# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data, data_slice
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import pickle

from contextlib import redirect_stdout

# Add code to load in the data.
data = pd.read_csv("../data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder = encoder, lb = lb
)

# Train and save a model.
model = train_model(X_train, y_train)
with open('../model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../model/onehotencoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('../model/labelbinarizer.pkl', 'wb') as f:
    pickle.dump(lb, f)
############################################
# Test the trained model on the test set
test_pred = inference(model, X_test)
test_precision, test_recall, test_fbeta = compute_model_metrics(test_pred, y_test)
print("Perisoin on the test set: {}".format(test_precision))
print("Recall rate on the test set: {}".format(test_recall))
print("F1 score on the test set: {}".format(test_fbeta))

############################################
def slice_pred(model, data, slice_feature):
    """ Run inferences on slices and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    data : pd.DataFrame
        Dataframe containing the features, at least columns in `categorical_features`.
    slice_feature: str
        The categorical feature used to slice the data.
    Returns
    -------
    sliced_results : dict
        Performance of the model for each subgroups of the slice feature.
    """

    sliced_data = data_slice(data, slice_feature=slice_feature)

    sliced_results = {}
    for key, slice in sliced_data.items():
        slice_test, slice_label, _, _ = process_data(slice, categorical_features=cat_features,
                             label="salary", training=False,
                             encoder = encoder, 
                             lb = lb
                             )
        slice_pred = inference(model, slice_test)
        slice_metrics = compute_model_metrics(slice_pred, slice_label)
        sliced_results[key] = slice_metrics
    return sliced_results
#############################################

# Try sliced prediction for each category features
with open('slice_output.txt', 'w') as f:
    with redirect_stdout(f):
        for feature in cat_features:
            print("*"*20)
            print(feature)
            sliced_results = slice_pred(model, data = test, slice_feature = feature)
            for key, results in sliced_results.items():
                print(key)
                print(results)