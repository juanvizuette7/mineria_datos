from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CONFIGS = {
    "dolar": {
        "dataset": "dolar_data.csv",
        "model": "modelo_dolar.pkl",
        "target": "Precio_Dolar",
        "features": ["Dia", "Inflacion", "Tasa_interes"],
        "metrics_label": "MSE / R2",
    },
    "glucosa": {
        "dataset": "glucosa_data.csv",
        "model": "modelo_glucosa.pkl",
        "target": "Nivel_Glucosa",
        "features": ["Edad", "IMC", "Actividad_Fisica"],
        "metrics_label": "MSE / R2",
    },
    "energia": {
        "dataset": "energia_data.csv",
        "model": "modelo_energia.pkl",
        "target": "Consumo_Energia",
        "features": ["Temperatura", "Hora", "Dia_Semana"],
        "metrics_label": "RMSE / R2",
    },
}


def evaluate_and_export(name: str, cfg: dict) -> dict:
    df = pd.read_csv(DATA_DIR / cfg["dataset"])
    X = df[cfg["features"]]
    y = df[cfg["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    standardized_model = LinearRegression().fit(X_scaled, y)
    standardized = {
        feature: float(value)
        for feature, value in zip(cfg["features"], standardized_model.coef_)
    }

    final_model = LinearRegression()
    final_model.fit(X, y)
    joblib.dump(final_model, MODELS_DIR / cfg["model"])

    return {
        "name": name,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "intercept": float(model.intercept_),
        "coefficients": {
            feature: float(value) for feature, value in zip(cfg["features"], model.coef_)
        },
        "standardized_coefficients": standardized,
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "final_intercept": float(final_model.intercept_),
        "final_coefficients": {
            feature: float(value)
            for feature, value in zip(cfg["features"], final_model.coef_)
        },
    }


def main() -> None:
    results = {
        name: evaluate_and_export(name, cfg) for name, cfg in CONFIGS.items()
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
