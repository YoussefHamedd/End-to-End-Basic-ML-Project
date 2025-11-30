# Hyper parameter tuning and model training by finding the best model



import os
import sys
import json
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            # Set MLflow experiment
            mlflow.set_experiment("student_performance_prediction")

            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # Start MLflow run
            with mlflow.start_run(run_name="model_training"):

                # Log dataset info
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", X_test.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])

                model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                 models=models,param=params)

                # Log all model scores
                for model_name, score in model_report.items():
                    mlflow.log_metric(f"{model_name}_r2_score", score)

                ## To get best model score from dict
                best_model_score = max(sorted(model_report.values()))

                ## To get best model name from dict

                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
                best_model = models[best_model_name]

                if best_model_score<0.6:
                    raise CustomException("No best model found")
                logging.info(f"Best found model on both training and testing dataset")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted=best_model.predict(X_test)

                r2_square = r2_score(y_test, predicted)
                mse = mean_squared_error(y_test, predicted)
                mae = mean_absolute_error(y_test, predicted)

                # Log best model metrics
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("best_model_r2_score", r2_square)
                mlflow.log_metric("best_model_mse", mse)
                mlflow.log_metric("best_model_mae", mae)

                # Log model
                mlflow.sklearn.log_model(best_model, "model")

                # Save metrics to file for DVC
                metrics = {
                    "r2_score": r2_square,
                    "mse": mse,
                    "mae": mae,
                    "best_model": best_model_name
                }
                with open("metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)

                mlflow.log_artifact("metrics.json")

                logging.info(f"MLflow tracking completed. R2 Score: {r2_square}")

                return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)