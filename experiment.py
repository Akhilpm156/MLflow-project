import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1score  = f1_score(actual, pred)
    return precision, recall, f1score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the Heart diseases csv file
    file = (
        "./data/hcd.csv"
    )
    try:
        data = pd.read_csv(file)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    
    # Removie None values
    data.dropna(inplace=True)
   

    x = data.drop(["TenYearCHD"], axis=1)
    
    y = data["TenYearCHD"]

    # Split the data into training and test sets. (0.75, 0.25) split.
    x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=10)

    # scale train and test data
   
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    

    C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = LogisticRegression(C, l1_ratio, random_state=42)
        lr.fit(x_train, y_train)

        predicted_qualities = lr.predict(x_test)

        (precision,recall,f1score) = eval_metrics(y_tes>

        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1score: {f1score}")

        mlflow.log_param("precision", precision)
        mlflow.log_param("recall", recall)
        mlflow.log_metric("f1score", f1score)

   
        # For remote server only (Dagshub)
        remote_server_uri = "https://dagshub.com/Akhilpm156/MLflow-project.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="LogisticRegression")
        else:
            mlflow.sklearn.log_model(lr, "model")
