"""Tests for hop history: hop, back, breadcrumb (docs/06-DASHBOARD.md 5.3)."""

from app.logic import navigation as nav


def test_opening_from_a_list_starts_fresh():
    state = {nav.SELECTED: "A", nav.HISTORY: ["X", "Y"]}
    nav.open_account(state, "B")
    assert state[nav.SELECTED] == "B" and state[nav.HISTORY] == []


def test_hop_remembers_where_we_came_from_and_back_returns():
    state = {}
    nav.open_account(state, "A")
    nav.hop_to(state, "B")
    nav.hop_to(state, "C")
    assert state[nav.SELECTED] == "C" and state[nav.HISTORY] == ["A", "B"]
    nav.go_back(state)
    assert state[nav.SELECTED] == "B" and state[nav.HISTORY] == ["A"]
    nav.go_back(state)
    nav.go_back(state)                      # nothing left: stays put
    assert state[nav.SELECTED] == "A" and state[nav.HISTORY] == []


def test_hopping_to_the_open_account_does_nothing():
    state = {}
    nav.open_account(state, "A")
    view = state[nav.VIEW]
    nav.hop_to(state, "A")
    assert state[nav.HISTORY] == [] and state[nav.VIEW] == view


def test_breadcrumb_shows_the_last_items_and_jump_trims_the_path():
    state = {}
    nav.open_account(state, "A")
    for account in "BCDEFGH":
        nav.hop_to(state, account)
    crumbs = nav.breadcrumb(state, 6)
    assert [account for _, account in crumbs] == list("CDEFGH")
    assert crumbs[-1][0] is None            # the open account is not clickable
    position_of_d = dict((account, position) for position, account in crumbs)["D"]
    nav.jump_back_to(state, position_of_d)
    assert state[nav.SELECTED] == "D" and state[nav.HISTORY] == list("ABC")


def test_every_move_gives_the_graph_a_new_identity():
    state = {}
    nav.open_account(state, "A")
    first = state[nav.VIEW]
    nav.hop_to(state, "B")
    nav.go_back(state)
    assert state[nav.VIEW] == first + 2


def test_clear_forgets_everything():
    state = {}
    nav.open_account(state, "A")
    nav.hop_to(state, "B")
    nav.clear(state)
    assert state[nav.SELECTED] is None and state[nav.HISTORY] == []
