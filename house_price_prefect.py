from prefect import flow, task
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@task
def load_dataset():

    print("Loading House Price Dataset...")

    df = pd.read_csv("house_prices_practice.csv")

    print("Dataset Loaded Successfully")
    print("Dataset Shape:", df.shape)

    return df


@task
def preprocess_data(df):

    print("\nChecking Missing Values:")

    print(df.isnull().sum())

    features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr",
        "LotArea"
    ]

    X = df[features]
    y = df["SalePrice"]

    print("\nFeatures:")
    print(features)

    print("\nInput Shape:")
    print(X.shape)

    print("\nTarget Shape:")
    print(y.shape)

    return X, y


@task
def train_model(X, y):

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\nModel Training Completed")

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R2 Score:", r2)

    return model, X_test, y_test


@task
def make_prediction(model, X_test, y_test):

    print("\nMaking House Price Prediction...")

    sample = X_test.iloc[[0]]

    actual = y_test.iloc[0]

    prediction = model.predict(sample)[0]

    print("\nInput Values:")
    print(sample.to_string(index=False))

    print("\nActual House Price:")
    print(actual)

    print("\nPredicted House Price:")
    print(prediction)

    return prediction


@flow
def house_price_prediction_workflow():

    print("========================================")
    print("HOUSE PRICE PREDICTION PREFECT WORKFLOW")
    print("========================================")

    df = load_dataset()

    X, y = preprocess_data(df)

    model, X_test, y_test = train_model(X, y)

    prediction = make_prediction(
        model,
        X_test,
        y_test
    )

    print("\n========================================")
    print("WORKFLOW COMPLETED")
    print("Final Predicted Price:", prediction)
    print("========================================")

    return prediction


if __name__ == "__main__":
    house_price_prediction_workflow()
