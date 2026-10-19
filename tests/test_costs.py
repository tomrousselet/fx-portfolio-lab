"""Tests of the transaction cost model (decision D14)."""

import pandas as pd
import pytest

from fxlab.costs import BASE_CASE, BP, CostModel, rebalancing_cost

MODEL = CostModel(outright={"AUD": 3 * BP, "JPY": 2 * BP}, swap={"AUD": 1 * BP, "JPY": 0.5 * BP})


def cost(old, new):
    return rebalancing_cost(pd.Series(old), pd.Series(new), MODEL)


def test_opening_is_outright():
    """From no position to 100 of AUD: 100 traded outright at 3 bp."""
    assert cost({"AUD": 0.0}, {"AUD": 100.0}) == pytest.approx(100 * 3 * BP)


def test_unchanged_position_pays_roll_only():
    """Keeping 100 of AUD: rolled with a swap at 1 bp, nothing traded outright."""
    assert cost({"AUD": 100.0}, {"AUD": 100.0}) == pytest.approx(100 * 1 * BP)


def test_increase_rolls_old_and_trades_difference():
    """100 -> 150: 100 rolled (1 bp) and 50 bought outright (3 bp)."""
    assert cost({"AUD": 100.0}, {"AUD": 150.0}) == pytest.approx(100 * 1 * BP + 50 * 3 * BP)


def test_decrease_rolls_new_and_trades_difference():
    """100 -> 40: 40 rolled and 60 sold outright."""
    assert cost({"AUD": 100.0}, {"AUD": 40.0}) == pytest.approx(40 * 1 * BP + 60 * 3 * BP)


def test_sign_change_is_close_and_reopen():
    """Long 100 -> short 50: nothing kept, 150 traded outright."""
    assert cost({"AUD": 100.0}, {"AUD": -50.0}) == pytest.approx(150 * 3 * BP)


def test_short_positions_symmetric():
    """Short 100 kept short 100: same roll cost as a long position."""
    assert cost({"JPY": -100.0}, {"JPY": -100.0}) == pytest.approx(100 * 0.5 * BP)


def test_costs_add_up_across_currencies():
    total = cost({"AUD": 0.0, "JPY": 100.0}, {"AUD": 100.0, "JPY": 100.0})
    assert total == pytest.approx(100 * 3 * BP + 100 * 0.5 * BP)


def test_scaling():
    assert BASE_CASE.scaled(2).outright["EUR"] == pytest.approx(2 * BASE_CASE.outright["EUR"])
    assert BASE_CASE.scaled(0).swap["NOK"] == 0


def test_every_g10_currency_has_a_cost():
    g9 = {"EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "NOK", "SEK"}
    assert set(BASE_CASE.outright) == g9 == set(BASE_CASE.swap)


def test_unknown_currency_raises():
    with pytest.raises(KeyError):
        cost({"XYZ": 0.0}, {"XYZ": 1.0})
