"""Valuation logic: the market model behaves sensibly and condition costs follow the survey."""
from datetime import date

import pytest

from inference.valuate import get_body_deduction, get_engine_deduction, predict_market_price, valuate


def market(assets, vehicle):
    model, meta, _, _ = assets
    return predict_market_price(model, meta, vehicle)["market_value_lkr"]


def test_more_mileage_never_raises_price(assets, alto):
    prices = [market(assets, {**alto, "mileage_km": km}) for km in range(40000, 240001, 20000)]
    assert prices == sorted(prices, reverse=True)
    assert prices[0] > prices[-1]


def test_newer_car_is_worth_more(assets, alto):
    prices = [market(assets, {**alto, "maf_year": y, "reg_year": y}) for y in range(2010, 2019)]
    assert prices == sorted(prices)


def test_more_owners_never_raises_price(assets, alto):
    prices = [market(assets, {**alto, "previous_owners": n}) for n in range(1, 9)]
    assert prices == sorted(prices, reverse=True)


def test_market_value_is_plausible_for_an_alto(assets, alto):
    assert 3_000_000 < market(assets, alto) < 5_000_000


def test_engine_deduction_follows_health_colour(assets):
    costs = assets[3]
    green = get_engine_deduction("healthy", 0.95, costs, mhs_score=95)
    amber = get_engine_deduction("tappet", 0.6, costs, mhs_score=60)
    red = get_engine_deduction("knocking", 0.9, costs, mhs_score=30)
    assert (green["health_color"], amber["health_color"], red["health_color"]) == ("green", "amber", "red")
    assert green["deduction_lkr"] == 0 < amber["deduction_lkr"] < red["deduction_lkr"]


def test_engine_uses_health_score_not_fault_name(assets):
    costs = assets[3]
    # Same score, different fault names -> same deduction
    a = get_engine_deduction("knocking", 0.7, costs, mhs_score=65)
    b = get_engine_deduction("tappet", 0.7, costs, mhs_score=65)
    assert a["deduction_lkr"] == b["deduction_lkr"]


def test_engine_falls_back_to_fault_class_without_score(assets):
    costs = assets[3]
    assert get_engine_deduction("healthy", 0.9, costs)["health_color"] == "green"
    assert get_engine_deduction("knocking", 0.9, costs)["health_color"] == "red"


@pytest.mark.parametrize("score, category", [(100, "none"), (95, "minor"), (75, "moderate"), (40, "severe")])
def test_body_bands(assets, score, category):
    assert get_body_deduction(score, assets[3])["damage_category"] == category


def test_worse_condition_lowers_fair_value(assets, alto):
    good = valuate(alto, 4.0, mhs_score=95, body_score=100, vin_status="original", preloaded_assets=assets)
    bad = valuate(alto, 4.0, mhs_score=30, body_score=40, vin_status="original", preloaded_assets=assets)
    assert bad["fair_value_lkr"] < good["fair_value_lkr"]
    assert bad["base_market_value_lkr"] == good["base_market_value_lkr"]   # the ML part ignores condition


def test_price_factors_explain_mileage(assets, alto):
    high = valuate({**alto, "mileage_km": 200000}, 4.0, vin_status="original", preloaded_assets=assets)
    mileage = [f for f in high["price_factors"] if f["factor"] == "mileage"]
    assert mileage and mileage[0]["impact_lkr"] < 0
    assert "200,000 km" in mileage[0]["text"]


def test_stale_market_data_is_flagged(assets, alto):
    fresh = valuate(alto, 4.0, vin_status="original", preloaded_assets=assets, today=date(2026, 11, 1))
    stale = valuate(alto, 4.0, vin_status="original", preloaded_assets=assets, today=date(2027, 6, 1))
    assert fresh["market_data_warning"] is None
    assert "Retrain" in stale["market_data_warning"]
