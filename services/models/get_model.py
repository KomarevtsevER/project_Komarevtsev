from pathlib import Path

import cloudpickle
import mlflow
import mlflow.sklearn


TRACKING_URI = "http://127.0.0.1:5000"

# Вставь значение, напечатанное новой ячейкой ноутбука.
RUN_ID = "506f7bd7007d451fa1414639a4c88f93"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.sklearn.load_model(
        f"runs:/{RUN_ID}/model"
    )

    output_path = Path(__file__).resolve().parent / "model.pkl"

    with output_path.open("wb") as file:
        cloudpickle.dump(model, file)

    print(f"Полный pipeline сохранён: {output_path}")


if __name__ == "__main__":
    main()