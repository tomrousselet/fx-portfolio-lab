"""Transaction costs of forward positions (paper, section 4.4; decisions D14-D15).

A forward position kept from one month to the next is rolled with an FX swap,
which costs the swap half-spread on the notional kept. Any change of position
(opening, closing, resizing, changing sign) is an outright trade, which costs
the outright half-spread on the notional traded.

Costs are proportional to notional, expressed as a fraction of notional
(1 basis point = 0.0001).
"""

from dataclasses import dataclass

import pandas as pd

BP = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Half-spreads per currency, as a fraction of notional."""

    outright: dict
    swap: dict

    def scaled(self, factor: float) -> "CostModel":
        """The same model with every half-spread multiplied by `factor` (sensitivity analysis)."""
        return CostModel(
            outright={k: v * factor for k, v in self.outright.items()},
            swap={k: v * factor for k, v in self.swap.items()},
        )


def _tiered(tiers: dict) -> dict:
    return {ccy: value for currencies, value in tiers.items() for ccy in currencies}


# Base case (decision D15): an ASSUMPTION by liquidity tier, not an observed spread.
# Results are reported for this base case scaled by 0, 1, 2 and 5, together with the
# break-even scale at which a strategy stops being profitable.
BASE_CASE = CostModel(
    outright=_tiered({
        ("EUR", "JPY", "GBP", "CHF", "CAD"): 2 * BP,
        ("AUD", "NZD"): 3 * BP,
        ("NOK", "SEK"): 5 * BP,
    }),
    swap=_tiered({
        ("EUR", "JPY", "GBP", "CHF", "CAD"): 1 * BP,
        ("AUD", "NZD"): 1.5 * BP,
        ("NOK", "SEK"): 2.5 * BP,
    }),
)

NO_COSTS = BASE_CASE.scaled(0.0)


def rebalancing_cost(old: pd.Series, new: pd.Series, model: CostModel) -> float:
    """Cost, in currency units, of moving from notionals `old` to notionals `new`.

    Per currency: the notional kept, min(|old|, |new|) when both have the same sign,
    is rolled at the swap half-spread; the notional traded, |new - old|, pays the
    outright half-spread. A sign change keeps nothing: the position is closed and
    reopened, so |new - old| = |old| + |new| is traded outright.
    """
    old, new = old.align(new, fill_value=0.0)
    old, new = old.fillna(0.0), new.fillna(0.0)
    same_sign = (old * new) > 0
    kept = old.abs().where(same_sign, 0.0).clip(upper=new.abs())
    traded = (new - old).abs()
    swap = pd.Series(model.swap).reindex(old.index)
    outright = pd.Series(model.outright).reindex(old.index)
    if swap.isna().any() or outright.isna().any():
        missing = list(old.index[swap.isna() | outright.isna()])
        raise KeyError(f"no transaction cost defined for {missing}")
    return float((kept * swap).sum() + (traded * outright).sum())
