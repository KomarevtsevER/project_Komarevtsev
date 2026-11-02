from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class FastAPIHandler:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "model.pkl"
            )

        with Path(model_path).open("rb") as file:
            self.model = cloudpickle.load(file)

        if not isinstance(self.model, Pipeline):
            raise TypeError(
                "В model.pkl должен находиться полный sklearn Pipeline. "
                "Обнови модель через get_model.py."
            )

    def predict(self, item_features: dict) -> int:
       
        item_df = pd.DataFrame([item_features])

        # Повторяем вычисления из ЛР1.
        item_df["total_pixels"] = (
            item_df["px_height"].astype("int32")
            * item_df["px_width"].astype("int32")
        )

        item_df["pixel_density"] = (
            item_df["total_pixels"]
            / (
                item_df["sc_h"].astype("float32")
                * item_df["sc_w"].astype("float32")
            )
        )

        item_df["battery_efficiency"] = (
            item_df["battery_power"].astype("float32")
            / item_df["mobile_wt"].astype("float32")
        )

        item_df["total_camera"] = (
            item_df["pc"] + item_df["fc"]
        )

        # Согласуем точность этих столбцов с данными ЛР2.
        item_df["clock_speed"] = item_df["clock_speed"].astype("float32")
        item_df["m_dep"] = item_df["m_dep"].astype("float32")

        
        predictions = self.model.predict(item_df)

        return int(np.asarray(predictions).item())