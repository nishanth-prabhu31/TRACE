"""Which account is open, and the hop history behind it (docs/06-DASHBOARD.md 5.3).

Every function works on a plain dict-like `state`, so it can be tested without Streamlit.
"""

from collections.abc import MutableMapping

SELECTED = "selected_account"
HISTORY = "hop_history"
VIEW = "graph_view"        # counts hops; gives the graph a fresh identity after each move


def _moved(state: MutableMapping) -> None:
    state[VIEW] = state.get(VIEW, 0) + 1


def open_account(state: MutableMapping, account_id: str) -> None:
    """Open an account from a list (alert queue, rings): starts a fresh history."""
    state[SELECTED] = account_id
    state[HISTORY] = []
    _moved(state)


def hop_to(state: MutableMapping, account_id: str) -> None:
    """Hop from the open account to a neighbour, remembering where we came from."""
    current = state.get(SELECTED)
    if account_id == current:
        return
    history = list(state.get(HISTORY, []))
    if current is not None:
        history.append(current)
    state[HISTORY] = history
    state[SELECTED] = account_id
    _moved(state)


def go_back(state: MutableMapping) -> None:
    """Return to the previous account; does nothing if there is no history."""
    history = list(state.get(HISTORY, []))
    if not history:
        return
    state[SELECTED] = history.pop()
    state[HISTORY] = history
    _moved(state)


def jump_back_to(state: MutableMapping, index: int) -> None:
    """Jump to an earlier breadcrumb item (by its position in the history) and trim the rest."""
    history = list(state.get(HISTORY, []))
    if not 0 <= index < len(history):
        return
    state[SELECTED] = history[index]
    state[HISTORY] = history[:index]
    _moved(state)


def clear(state: MutableMapping) -> None:
    """Forget the open account and its history (used when the run changes)."""
    state[SELECTED] = None
    state[HISTORY] = []
    _moved(state)


def breadcrumb(state: MutableMapping, length: int) -> list[tuple[int | None, str]]:
    """The last `length` accounts on the path: (history position or None for the open one, id)."""
    history = list(state.get(HISTORY, []))
    items: list[tuple[int | None, str]] = list(enumerate(history))
    if state.get(SELECTED) is not None:
        items.append((None, state[SELECTED]))
    return items[-length:]
