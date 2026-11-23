"""Number formatting shared by the pipeline (summaries) and the app."""

LAKH, CRORE = 100_000, 10_000_000


def indian_grouping(whole: int) -> str:
    """Group digits the Indian way: 125000 -> '1,25,000'."""
    digits = str(abs(whole))
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    pairs = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *pairs, tail])


def format_inr(rupees: float, short: bool = False) -> str:
    """Rupee text: '₹1,25,000', or with short=True '₹4.35 L' / '₹1.2 Cr' for large amounts."""
    if short and abs(rupees) >= CRORE * 1000:              # very large: whole crores, grouped
        return f"₹{indian_grouping(round(rupees / CRORE))} Cr"
    if short and abs(rupees) >= CRORE:
        return f"₹{rupees / CRORE:.2f}".rstrip("0").rstrip(".") + " Cr"
    if short and abs(rupees) >= LAKH:
        return f"₹{rupees / LAKH:.2f}".rstrip("0").rstrip(".") + " L"
    sign = "-" if rupees < 0 else ""
    return f"{sign}₹{indian_grouping(round(rupees))}"
