"""Turn transactions into per-account events, the shared input of every feature family.

Each transaction becomes two events: an OUT event for the sender and an IN event for
the receiver. Feature code then groups events by account.
"""

import pandas as pd


def account_events(transactions: pd.DataFrame, cutoff: pd.Timestamp,
                   accounts: pd.Index | None = None) -> pd.DataFrame:
    """Events with timestamp <= cutoff, optionally only for the given accounts.

    Columns: account_id, counterparty, direction ("IN"/"OUT"), timestamp, amount,
    amount_original, channel, is_self.
    """
    txns = transactions[transactions["timestamp"] <= cutoff]
    keep = ["timestamp", "amount", "amount_original", "channel", "is_self"]
    out = txns[keep].assign(account_id=txns["sender_id"], counterparty=txns["receiver_id"], direction="OUT")
    inc = txns[keep].assign(account_id=txns["receiver_id"], counterparty=txns["sender_id"], direction="IN")
    events = pd.concat([out, inc], ignore_index=True)
    if accounts is not None:
        events = events[events["account_id"].isin(accounts)]
    return events.sort_values(["account_id", "timestamp"], kind="stable").reset_index(drop=True)


def active_accounts(transactions: pd.DataFrame, cutoff: pd.Timestamp, lookback_hours: int) -> pd.Index:
    """Accounts with at least one transaction in (cutoff - lookback, cutoff]."""
    start = cutoff - pd.Timedelta(hours=lookback_hours)
    window = transactions[(transactions["timestamp"] > start) & (transactions["timestamp"] <= cutoff)]
    return pd.Index(pd.concat([window["sender_id"], window["receiver_id"]]).unique(), name="account_id")
