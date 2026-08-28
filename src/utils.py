import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():

            model_params = param.get(model_name, {})

            # If model has parameters for tuning
            if model_params:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

            else:
                # For models like LinearRegression with no parameters
                model.fit(X_train, y_train)
                best_model = model

            # Store fitted model back into dictionary
            models[model_name] = best_model

            y_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test, y_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)