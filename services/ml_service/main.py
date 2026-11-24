from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from api_handler import FastAPIHandler


class PhoneFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    battery_power: int = Field(gt=0)
    blue: Literal[0, 1]
    clock_speed: float = Field(gt=0, allow_inf_nan=False)
    dual_sim: Literal[0, 1]
    fc: int = Field(ge=0)
    four_g: Literal[0, 1]
    int_memory: int = Field(gt=0)
    m_dep: float = Field(gt=0, allow_inf_nan=False)
    mobile_wt: int = Field(gt=0)
    n_cores: int = Field(ge=1, le=8)
    pc: int = Field(ge=0)
    px_height: int = Field(ge=0)
    px_width: int = Field(gt=0)
    ram: int = Field(gt=0)
    sc_h: int = Field(gt=0)
    sc_w: int = Field(gt=0)
    talk_time: int = Field(gt=0)
    three_g: Literal[0, 1]
    touch_screen: Literal[0, 1]
    wifi: Literal[0, 1]


prediction_histogram = Histogram(
    "mobile_price_prediction",
    "Distribution of predicted mobile phone price classes",
    buckets=(0.5, 1.5, 2.5, 3.5),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.handler = FastAPIHandler()
    yield


app = FastAPI(lifespan=lifespan)


instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=[
        r"^/metrics$",
        r"^/$",
    ],
)

instrumentator.instrument(app).expose(
    app,
    include_in_schema=False,
)


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/api/prediction")
def make_prediction(item_id: int, item_features: PhoneFeatures):
    prediction = app.state.handler.predict(
        item_features.model_dump()
    )

    prediction_histogram.observe(prediction)

    return {
        "item_id": item_id,
        "predict": prediction,
    }