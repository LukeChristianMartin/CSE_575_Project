import re
import openml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_openml_df_from_link(openML_link):

    # Extract the dataset ID using regular expressions
    match = re.search(r'/d/(\d+)', openML_link)

    if match:
        dataset_id = int(match.group(1))
    else:
        raise ValueError("Invalid OpenML link format")

    # Go to openML and download the dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Convert dataset to pandas dataframe
    df_features = pd.DataFrame(data=X)
    df_classification = pd.DataFrame(data=y)
    df_classification = df_classification.to_numpy().ravel()

    return df_features, df_classification


def cross_validate_model_with_random_search(model_type, param_distributions, num_of_searches):

    # Create a Randomized Search Cross Validation object
    model_search = RandomizedSearchCV(model_type, param_distributions=param_distributions,
                                      n_iter=num_of_searches, cv=5,
                                      random_state=42, n_jobs=-1)

    # Perform the random search for hyperparameter tuning
    model_search.fit(X_train, y_train)

    # Print the best hyperparameters and corresponding accuracy
    best_params = model_search.best_params_
    best_accuracy = model_search.best_score_

    print("Best Hyperparameters:")
    print(best_params)
    print("Best Accuracy:", best_accuracy)

    best_model_from_search = model_search.best_estimator_

    return best_model_from_search


def evaluate_model_on_test_dataset(model, x_test, y_test):
    y_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", test_accuracy)



if __name__ == '__main__':

    # Replace this with your OpenML dataset link
    openml_link = "https://www.openml.org/d/44123"

    openML_features, openML_classification = get_openml_df_from_link(openml_link)

    # Split the data into test and training data
    X_train, X_test, y_train, y_test = train_test_split(openML_features, openML_classification, test_size=0.3, random_state=42)

    rf_classifier = RandomForestClassifier(random_state=42)

    # The hyperparameters for Random Forest
    rf_hyperparameters_dist = {
        'n_estimators': np.arange(50, 200, 10),
        'max_depth': np.arange(1, 20),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'bootstrap': [True, False]
    }

    best_model = cross_validate_model_with_random_search(rf_classifier, rf_hyperparameters_dist, 3)

    evaluate_model_on_test_dataset(best_model, X_test, y_test)

