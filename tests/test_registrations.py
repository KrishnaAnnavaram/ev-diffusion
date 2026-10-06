"""Reference problem 4: a changing mix of states and national rows."""

import pandas as pd
import pytest

from ev_diffusion.data import synthetic
from ev_diffusion.data.registrations import (
    best_balanced_window,
    build_series,
    coverage_summary,
    coverage_table,
    is_partial_last_year,
    load_registrations,
)
from ev_diffusion.schema import SchemaError


def test_national_rows_are_never_added_to_state_rows(registrations):
    series = build_series(registrations, policy="national")
    national = registrations[registrations["state"] == "US"].groupby("year")["count"].sum()
    for year, value in zip(series.years, series.counts):
        assert value == pytest.approx(national[year])
    naive = registrations.groupby("year")["count"].sum()
    assert (naive.loc[series.years].to_numpy() > series.counts).all()


def test_balanced_panel_uses_only_states_present_in_every_year(registrations):
    series = build_series(registrations, policy="balanced")
    states = registrations[registrations["state"] != "US"]
    for state in series.states:
        present = set(states[states["state"] == state]["year"])
        assert set(series.years) <= present
    late = [s for s, y in synthetic.ENTRY_YEAR.items() if y > series.years[0]]
    assert not set(late) & set(series.states)


def test_best_window_maximises_state_years():
    frame = pd.DataFrame(
        [{"state": s, "year": y, "count": 1.0} for s in ("A", "B") for y in range(2010, 2020)]
        + [{"state": s, "year": y, "count": 1.0} for s in ("C", "D", "E") for y in range(2014, 2020)]
    )
    start, states = best_balanced_window(coverage_table(frame), 2019, min_years=4)
    assert start == 2014 and states == ["A", "B", "C", "D", "E"]


def test_partial_last_year_is_detected_and_removed(registrations):
    series = build_series(registrations)
    assert series.years[-1] == synthetic.YEARS[-2]
    assert any("looks partial" in note for note in series.notes)
    assert not is_partial_last_year(pd.Series([10.0, 20.0, 30.0, 40.0, 38.0], index=range(5)))


def test_last_complete_year_overrides_the_heuristic(registrations):
    series = build_series(registrations, last_complete_year=2020)
    assert series.years[-1] == 2020


def test_coverage_summary_shows_entry_years(registrations):
    summary = coverage_summary(registrations).set_index("state")
    for state, year in synthetic.ENTRY_YEAR.items():
        assert summary.loc[state, "first_year"] == year
    assert bool(summary.loc["US", "national"])


def test_prototype_headers_are_accepted_and_missing_state_is_noted(tmp_path):
    path = tmp_path / "old.csv"
    rows = [{"registration_year": y, "vehicle_make_&_model": "X", "vehicle_count": 10 * (y - 2009)} for y in range(2010, 2020)]
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = load_registrations(path)
    series = build_series(frame)
    assert any("no state column" in note for note in series.notes)
    assert series.counts[0] == 10


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"year": 2010}], "missing required column 'count'"),
        ([{"year": 2010, "count": "many"}], "not numeric"),
        ([{"year": 2010, "count": -5}], "below 0"),
    ],
)
def test_schema_errors_are_clear(tmp_path, rows, message):
    path = tmp_path / "bad.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(SchemaError, match=message):
        load_registrations(path)


def test_unknown_policy_is_refused(registrations):
    with pytest.raises(ValueError):
        build_series(registrations, policy="everything")
