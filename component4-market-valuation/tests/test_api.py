"""API tests: input handling, every verdict, VIN effect, scope refusal."""
import pytest


def post(client, auth, body):
    return client.post("/api/v1/valuate", json=body, headers=auth)


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.get_json()["models_loaded"] is True


def test_requires_token(client, alto):
    assert client.post("/api/v1/valuate", json={**alto, "listed_price_million": 4}).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert client.post("/api/v1/valuate", json={**alto, "listed_price_million": 4}, headers=bad).status_code == 401


@pytest.mark.parametrize("missing", ["maf_year", "mileage_km", "previous_owners", "listed_price_million"])
def test_missing_field_is_400(client, auth, alto, missing):
    body = {**alto, "listed_price_million": 4.0}
    body.pop(missing)
    r = post(client, auth, body)
    assert r.status_code == 400
    assert missing in r.get_json()["error"]


@pytest.mark.parametrize("change", [
    {"reg_year": 2014},                 # registered before it was made
    {"maf_year": 2099, "reg_year": 2099},
    {"mileage_km": "lots"},
    {"listed_price_million": -1},
])
def test_invalid_values_are_400(client, auth, alto, change):
    assert post(client, auth, {**alto, "listed_price_million": 4.0, **change}).status_code == 400


def test_non_json_is_400(client, auth):
    assert client.post("/api/v1/valuate", data="x", headers=auth).status_code == 400


@pytest.mark.parametrize("price, verdict", [(3.0, "GOOD_DEAL"), (4.0, "FAIR_PRICE"), (5.0, "OVERPRICED")])
def test_each_verdict(client, auth, alto, price, verdict):
    r = post(client, auth, {**alto, "listed_price_million": price, "vin_status": "original"})
    assert r.status_code == 200
    assert r.get_json()["verdict"] == verdict


def test_response_ranges_are_consistent(client, auth, alto):
    j = post(client, auth, {**alto, "listed_price_million": 4.3, "mhs_score": 40, "body_score": 70}).get_json()
    assert j["fair_low_lkr"] <= j["fair_value_lkr"] <= j["fair_high_lkr"]
    assert j["negotiation_min_lkr"] <= j["recommended_offer_lkr"] <= j["negotiation_max_lkr"] <= j["listed_price_lkr"]
    assert j["fair_value_lkr"] == pytest.approx(j["base_market_value_lkr"] - j["condition_deduction_lkr"], abs=1000)
    assert isinstance(j["price_factors"], list)


def test_altered_vin_blocks_price(client, auth, alto):
    j = post(client, auth, {**alto, "listed_price_million": 3.0, "vin_status": "altered"}).get_json()
    assert j["verdict"] == "CHECK_VIN"
    assert "issue with this vehicle's VIN" in j["verdict_message"]
    assert "do not" not in j["verdict_message"].lower()
    assert "fair_value_lkr" not in j


def test_vin_needs_review_discounts_and_never_good_deal(client, auth, alto):
    ok = post(client, auth, {**alto, "listed_price_million": 3.0, "vin_status": "original"}).get_json()
    review = post(client, auth, {**alto, "listed_price_million": 3.0, "vin_status": "needs review"}).get_json()
    assert ok["verdict"] == "GOOD_DEAL"
    assert review["verdict"] != "GOOD_DEAL"
    assert review["vin_discount_lkr"] > 0
    assert review["fair_value_lkr"] < ok["fair_value_lkr"]
    assert review["vin_warning"]


def test_unchecked_vin_warns_without_discount(client, auth, alto):
    j = post(client, auth, {**alto, "listed_price_million": 4.0}).get_json()   # no vin_status sent
    assert j["vin_status"] == "unknown"
    assert j["vin_discount_lkr"] == 0
    assert j["vin_warning"]


@pytest.mark.parametrize("change", [
    {"maf_year": 2005, "reg_year": 2005},
    {"mileage_km": 400000},
    {"previous_owners": 12},
    {"make": "Toyota", "model": "Vitz"},
    {"make": "Suzuki", "model": "Swift"},
])
def test_out_of_scope_is_refused(client, auth, alto, change):
    r = post(client, auth, {**alto, "listed_price_million": 4.0, **change})
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "out_of_scope"
    assert j["verdict"] is None
    assert "can't be valued reliably" in j["message"]


def test_alto_make_and_model_accepted(client, auth, alto):
    j = post(client, auth, {**alto, "listed_price_million": 4.0, "make": "Suzuki", "model": "Alto"}).get_json()
    assert j["status"] == "success"
