"""Turn numbers into the text shown on screen (docs/06-DASHBOARD.md section 2)."""

from trace_aml.config import load_settings
from trace_aml.formatting import format_inr, indian_grouping  # noqa: F401  (re-exported for pages and tests)

VERDICT_TEXT = {"LIKELY_MULE": "Likely mule", "WATCH": "Watch", "NORMAL": "Looks normal"}
DECISION_TEXT = {"CONFIRMED_MULE": "Confirmed mule", "FALSE_POSITIVE": "False positive",
                 "NEEDS_INVESTIGATION": "Needs investigation"}
def format_money(base_amount: float, currency: str, short: bool = False) -> str:
    """Show a base-currency amount in the chosen currency ('INR' or the base currency)."""
    settings = load_settings()["currency"]
    if currency == settings["display"]:
        return format_inr(base_amount * settings["display_units_per_base"], short)
    return f"{settings['base']} {base_amount:,.2f}"
