import argparse
import os
import random
import time

import httpx


def make_phone(rng: random.Random) -> dict:
    """Создать случайный объект с 20 признаками телефона."""

    pc = rng.randint(0, 20)
    four_g = rng.randint(0, 1)
    sc_h = rng.randint(5, 19)
    px_width = rng.randint(500, 1998)

    return {
        "battery_power": rng.randint(500, 1999),
        "blue": rng.randint(0, 1),
        "clock_speed": round(rng.uniform(0.5, 3.0), 1),
        "dual_sim": rng.randint(0, 1),
        "fc": rng.randint(0, pc),
        "four_g": four_g,
        "int_memory": rng.randint(2, 64),
        "m_dep": round(rng.uniform(0.1, 1.0), 1),
        "mobile_wt": rng.randint(80, 200),
        "n_cores": rng.randint(1, 8),
        "pc": pc,
        "px_height": rng.randint(100, px_width),
        "px_width": px_width,
        "ram": rng.randint(256, 3998),
        "sc_h": sc_h,
        "sc_w": rng.randint(4, min(18, sc_h)),
        "talk_time": rng.randint(2, 20),
        "three_g": 1 if four_g else rng.randint(0, 1),
        "touch_screen": rng.randint(0, 1),
        "wifi": rng.randint(0, 1),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=os.getenv(
            "API_URL",
            "http://ml_service:8000/api/prediction",
        ),
        help="Адрес endpoint-а предсказаний",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Количество запросов; 0 — работать до остановки",
    )

    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count не может быть отрицательным")

    rng = random.Random(42)
    item_id = 1

    print(f"Отправляем запросы на {args.url}", flush=True)

    with httpx.Client(timeout=10.0) as client:
        while args.count == 0 or item_id <= args.count:
            features = make_phone(rng)

            try:
                response = client.post(
                    args.url,
                    params={"item_id": item_id},
                    json=features,
                )

                response.raise_for_status()

                print(
                    f"item_id={item_id}, "
                    f"HTTP={response.status_code}, "
                    f"ответ={response.json()}",
                    flush=True,
                )

            except httpx.HTTPStatusError as error:
                print(
                    f"item_id={item_id}, "
                    f"HTTP={error.response.status_code}, "
                    f"ошибка={error.response.text}",
                    flush=True,
                )

            except httpx.RequestError as error:
                print(
                    f"item_id={item_id}, "
                    f"ошибка соединения: {error}",
                    flush=True,
                )

            item_id += 1

            if args.count != 0 and item_id > args.count:
                break

            time.sleep(rng.uniform(0, 5))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nГенератор остановлен.", flush=True)