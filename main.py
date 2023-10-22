import re
import random
import openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def get_openml_df_from_link(openML_link):

    # Extract the dataset ID using regular expressions
    match = re.search(r'/d/(\d+)', openML_link)

    label_encoder = LabelEncoder()

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
    df_classification = label_encoder.fit_transform(df_classification)

    return dataset_id, df_features, df_classification


def cross_validate_model_with_random_search(model_type, param_distributions, num_of_searches):

    # Create a Randomized Search Cross Validation object
    model_search = RandomizedSearchCV(estimator=model_type, param_distributions=param_distributions,
                                      n_iter=num_of_searches, cv=5,
                                      random_state=random.randint(1, 1000), n_jobs=-1)

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

    return test_accuracy

def initialize_rf_and_parameters():
    rf_classifier = RandomForestClassifier(random_state=random.randint(1, 1000))

    # The hyperparameters for Random Forest
    rf_hyperparameters_dist = {
        'n_estimators': np.arange(10, 200, 5),
        'max_depth': np.arange(1, 40, 2),
        'min_samples_split': np.arange(2, 20),
        'min_samples_leaf': np.arange(1, 20),
        'bootstrap': [True, False]
    }

    return rf_classifier, rf_hyperparameters_dist


def initialize_xgboost_and_parameters():
    xgb_classifier = XGBClassifier(random_state=random.randint(1, 1000))

    # The hyperparameters for XGBoost
    xgb_hyperparameters_dist = {
        'n_estimators': np.arange(10, 200, 5),
        'max_depth': np.arange(1, 40, 2),
        'min_child_weight': np.arange(1, 20),
        'gamma': np.arange(0, 5),
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.5, 1.0, 0.1),
        'learning_rate': [0.01, 0.1, 0.2, 0.3]
    }

    return xgb_classifier, xgb_hyperparameters_dist

def initialize_gbt_and_parameters():
    gbt_classifier = GradientBoostingClassifier(random_state=random.randint(1, 1000))

    # The hyperparameters for Gradient Boosting Tree
    gbt_hyperparameters_dist = {
        'n_estimators': np.arange(10, 200, 5),
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': np.arange(1, 40, 2),
        'min_samples_split': np.arange(2, 40),
        'min_samples_leaf': np.arange(1, 40),
    }

    return gbt_classifier, gbt_hyperparameters_dist


if __name__ == '__main__':

    user_openml_link = input("Enter link from paper: ")
    id_dataset, openML_features, openML_classification = get_openml_df_from_link(user_openml_link)

    # Split the data into test and training data
    X_train, X_test, y_train, y_test = train_test_split(openML_features, openML_classification, test_size=0.3, random_state=random.randint(1, 1000))

    classifier_type = None
    classifier_name = None
    classifier_hyper_parameters = None

    #xg for xg boost
    user_model_selection = input("Enter 'rf' or 'xg' or gbt: ")

    if user_model_selection.lower() == 'rf':
        classifier_type, classifier_hyper_parameters = initialize_rf_and_parameters()
        classifier_name = "Random Forest"

    elif user_model_selection.lower() == 'xg':
        classifier_type, classifier_hyper_parameters = initialize_xgboost_and_parameters()
        classifier_name = "XGBoost"

    elif user_model_selection.lower() == 'gbt':
        classifier_type, classifier_hyper_parameters = initialize_gbt_and_parameters()
        classifier_name = "GradientBoosting Trees"

    number_of_random_searches = input("Enter the number of random searches: ")
    number_of_random_searches = int(number_of_random_searches)

    list_of_model_accuracy = []
    num_rand_walks = [i for i in range(1, number_of_random_searches + 1)]

    for i in range(1, number_of_random_searches + 1):
        print("Iteration: " + str(i))
        best_model = cross_validate_model_with_random_search(classifier_type, classifier_hyper_parameters, i)

        list_of_model_accuracy.append(evaluate_model_on_test_dataset(best_model, X_test, y_test))

        print("\n\n")
    plt.semilogx(num_rand_walks, list_of_model_accuracy, label='Data Points', color='blue', marker='o')

    # Add labels and a title
    plt.xlabel('# of Random Search Iterations (Log Scale)')
    plt.ylabel('Test Accuracy')
    plt.title('Dataset ID: ' + str(id_dataset) + '      Classifier: ' + str(classifier_name))

    # Add a legend (if needed)
    plt.legend()

    # Show the plot
    plt.show()

