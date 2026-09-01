# Running Watinakama.LK Locally

This is the single source of truth for setting up and running the whole system: four independent Python backends plus the Flutter mobile app. Component-level READMEs still cover purpose, model details, and folder structure — this file only covers **how to get it running**.

## System overview

| # | Component | Directory | Framework | Port | Auth required |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | VIN Authentication | `component1-vin-authentication/` | FastAPI | `8000` | No |
| 2 | Body Condition | `component2-body-condition/` | FastAPI | `8080` | No |
| 3 | Engine Audio | `component3-engine-audio/` | Flask | `5003` | Yes — Bearer token |
| 4 | Market Valuation | `component4-market-valuation/` | Flask | `5004` | Yes — Bearer token |
| 5 | Mobile App | `watinakama_app/` | Flutter | — | — |

Each backend is fully independent, with its own virtual environment and `requirements.txt`. There is no shared root environment. Trained model files are already committed in each component, so a fresh clone can serve predictions immediately — none of the training/data-download scripts are required just to run the API.

**Prerequisites:** Python 3.9+, Flutter SDK `^3.11.5`.

---

## 1. Component 1 — VIN Authentication (port 8000)

```bash
cd component1-vin-authentication
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Docs / test UI: http://localhost:8000/docs

## 2. Component 2 — Body Condition (port 8080)

```bash
cd component2-body-condition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

> **Do not run `python main.py` directly.** Its `__main__` block hardcodes port `8000`, which collides with Component 1. Always start it via the `uvicorn main:app --port 8080` command above.

A `.env` (copy `.env.example`) with Roboflow credentials is only needed if you plan to run `download_dataset.py` or retrain — not for serving predictions.

Docs / test UI: http://localhost:8080/docs

## 3. Component 3 — Engine Audio (port 5003)

```bash
cd component3-engine-audio
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env       # Windows; `cp .env.example .env` on macOS/Linux
python api/app.py
```

Protected endpoints require an `Authorization: Bearer <API_SECRET_TOKEN>` header, where `API_SECRET_TOKEN` comes from `.env` (defaults to `dev-token-change-in-production` if unset).

## 4. Component 4 — Market Valuation (port 5004)

```bash
cd component4-market-valuation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env       # Windows; `cp .env.example .env` on macOS/Linux
python api/app.py
```

Same Bearer-token auth as Component 3. Docs / test UI: http://localhost:5004/apidocs/

## 5. Mobile App (Flutter)

```bash
cd watinakama_app
flutter pub get
```

Open [lib/services/api_service.dart](watinakama_app/lib/services/api_service.dart) and set `baseIp` to your development machine's **LAN IP address** (e.g. `192.168.1.10`) — not `127.0.0.1` or `localhost`, since the app needs to reach it from a phone or emulator over the network.

```bash
flutter run
```

The app pings all four backend ports on startup to confirm they're reachable before letting you begin an inspection, so start all four backends first.

---

## Troubleshooting

- **Port 8000 already in use / VIN results look like body-condition results** — Component 2 was probably started with `python main.py` instead of the `uvicorn` command above.
- **"model file not found" warning on Component 2 startup** — the `.h5` model files must sit directly inside `component2-body-condition/`, not a subfolder.
- **401 Unauthorized from Component 3 or 4** — the `Authorization` header's token doesn't match `API_SECRET_TOKEN` in that component's `.env`.
- **App can't reach any backend** — phone/emulator and dev machine must be on the same network; double-check `baseIp`, and make sure your firewall allows inbound connections on `8000`, `8080`, `5003`, `5004`.
