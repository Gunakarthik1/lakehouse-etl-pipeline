"""
Tests for the Quality Gate rules.

Each rule class is tested independently, followed by integration tests
of QualityGate.validate() with the default rule set.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from pipeline.quality import (
    CategoricalCheck,
    DateRangeCheck,
    NonNullCheck,
    QualityGate,
    QualityReport,
    RangeCheck,
    TypeCheck,
    UniquenessCheck,
    default_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Convenience: dict of lists → DataFrame."""
    return pd.DataFrame(kwargs)


def _apply(rule, df) -> list[bool]:
    return rule.apply(df).tolist()


# ---------------------------------------------------------------------------
# NonNullCheck
# ---------------------------------------------------------------------------

class TestNonNullCheck:
    def test_passes_all_present(self):
        df = _make_df(a=["x", "y"], b=[1, 2])
        result = _apply(NonNullCheck(["a", "b"]), df)
        assert result == [True, True]

    def test_fails_null_value(self):
        df = _make_df(a=["x", None], b=[1, 2])
        result = _apply(NonNullCheck(["a"]), df)
        assert result == [True, False]

    def test_fails_multiple_null_cols(self):
        df = _make_df(a=[None, "y"], b=["ok", None])
        # Row 0: a is null → fail. Row 1: b is null → fail.
        result = _apply(NonNullCheck(["a", "b"]), df)
        assert result == [False, False]

    def test_ignores_missing_column(self):
        """If the column is absent from the DataFrame, the check is skipped."""
        df = _make_df(a=["x", "y"])
        result = _apply(NonNullCheck(["z"]), df)  # column z does not exist
        assert result == [True, True]

    def test_passes_empty_col_list(self):
        df = _make_df(a=["x"])
        result = _apply(NonNullCheck([]), df)
        assert result == [True]


# ---------------------------------------------------------------------------
# RangeCheck
# ---------------------------------------------------------------------------

class TestRangeCheck:
    def test_passes_in_range(self):
        df = _make_df(price=[10.0, 50.0, 99.99])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [True, True, True]

    def test_fails_negative_value(self):
        df = _make_df(price=[-5.0, 10.0])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [False, True]

    def test_fails_above_max(self):
        df = _make_df(price=[99.99, 100.01])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [True, False]

    def test_fails_non_numeric_string(self):
        df = _make_df(price=["not_a_number", 10.0])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [False, True]

    def test_fails_none_value(self):
        df = _make_df(price=[None, 10.0])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [False, True]

    def test_boundary_inclusive(self):
        df = _make_df(price=[0.01, 100.0])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [True, True]

    def test_missing_column_passes_all(self):
        df = _make_df(other=[1, 2])
        result = _apply(RangeCheck("price", 0.01, 100.0), df)
        assert result == [True, True]


# ---------------------------------------------------------------------------
# TypeCheck
# ---------------------------------------------------------------------------

class TestTypeCheck:
    def test_passes_castable_to_float(self):
        df = _make_df(price=["9.99", 10.0, "100"])
        result = _apply(TypeCheck("price", "float"), df)
        assert result == [True, True, True]

    def test_fails_non_numeric_string(self):
        df = _make_df(price=["abc", 10.0])
        result = _apply(TypeCheck("price", "float"), df)
        assert result == [False, True]

    def test_fails_none(self):
        df = _make_df(price=[None, 5.0])
        result = _apply(TypeCheck("price", "float"), df)
        assert result == [False, True]

    def test_passes_castable_to_int(self):
        df = _make_df(qty=["3", 5, "10"])
        result = _apply(TypeCheck("qty", "int"), df)
        assert result == [True, True, True]

    def test_fails_float_string_for_int(self):
        # "3.5" is not directly castable to int
        df = _make_df(qty=["3.5"])
        result = _apply(TypeCheck("qty", "int"), df)
        assert result == [False]

    def test_missing_column_passes_all(self):
        df = _make_df(other=[1])
        result = _apply(TypeCheck("price", "float"), df)
        assert result == [True]


# ---------------------------------------------------------------------------
# DateRangeCheck
# ---------------------------------------------------------------------------

class TestDateRangeCheck:
    _min = datetime(2020, 1, 1, tzinfo=timezone.utc)
    _max = datetime.now(timezone.utc)

    def test_passes_valid_date(self):
        df = _make_df(ts=["2023-06-01T12:00:00+00:00"])
        result = _apply(DateRangeCheck("ts", self._min, self._max), df)
        assert result == [True]

    def test_fails_future_date(self):
        future = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
        df = _make_df(ts=[future])
        result = _apply(DateRangeCheck("ts", self._min, self._max), df)
        assert result == [False]

    def test_fails_ancient_date(self):
        df = _make_df(ts=["1990-01-01T00:00:00+00:00"])
        result = _apply(DateRangeCheck("ts", self._min, self._max), df)
        assert result == [False]

    def test_fails_null_timestamp(self):
        df = _make_df(ts=[None])
        result = _apply(DateRangeCheck("ts", self._min, self._max), df)
        assert result == [False]

    def test_fails_unparseable_timestamp(self):
        df = _make_df(ts=["not_a_date"])
        result = _apply(DateRangeCheck("ts", self._min, self._max), df)
        assert result == [False]

    def test_missing_column_passes_all(self):
        df = _make_df(other=["x"])
        result = _apply(DateRangeCheck("ts", self._min, self._max), df)
        assert result == [True]


# ---------------------------------------------------------------------------
# UniquenessCheck
# ---------------------------------------------------------------------------

class TestUniquenessCheck:
    def test_passes_all_unique(self):
        df = _make_df(event_id=["a", "b", "c"])
        result = _apply(UniquenessCheck("event_id"), df)
        assert result == [True, True, True]

    def test_fails_duplicate(self):
        df = _make_df(event_id=["a", "b", "a"])
        result = _apply(UniquenessCheck("event_id"), df)
        # First occurrence of "a" passes, second fails
        assert result[0] is True
        assert result[2] is False

    def test_keeps_first_occurrence(self):
        df = _make_df(event_id=["dup", "dup", "unique"])
        result = _apply(UniquenessCheck("event_id"), df)
        assert result == [True, False, True]

    def test_missing_column_passes_all(self):
        df = _make_df(other=["x"])
        result = _apply(UniquenessCheck("event_id"), df)
        assert result == [True]


# ---------------------------------------------------------------------------
# CategoricalCheck
# ---------------------------------------------------------------------------

class TestCategoricalCheck:
    VALID = ["page_view", "add_to_cart", "purchase", "search", "review"]

    def test_passes_valid_value(self):
        df = _make_df(event_type=["purchase", "page_view"])
        result = _apply(CategoricalCheck("event_type", self.VALID), df)
        assert result == [True, True]

    def test_fails_invalid_value(self):
        df = _make_df(event_type=["click", "purchase"])
        result = _apply(CategoricalCheck("event_type", self.VALID), df)
        assert result == [False, True]

    def test_fails_empty_string(self):
        df = _make_df(event_type=["", "purchase"])
        result = _apply(CategoricalCheck("event_type", self.VALID), df)
        assert result == [False, True]

    def test_fails_none(self):
        df = _make_df(event_type=[None, "search"])
        result = _apply(CategoricalCheck("event_type", self.VALID), df)
        assert result == [False, True]

    def test_missing_column_passes_all(self):
        df = _make_df(other=["x"])
        result = _apply(CategoricalCheck("event_type", self.VALID), df)
        assert result == [True]


# ---------------------------------------------------------------------------
# QualityGate integration
# ---------------------------------------------------------------------------

def _make_good_row(**overrides) -> dict:
    base = {
        "event_id": "evt_unique_001",
        "user_id": "user_1001",
        "session_id": "sess_abc",
        "event_type": "purchase",
        "timestamp": "2024-06-15T10:00:00+00:00",
        "product_id": "prod_001",
        "product_name": "Widget",
        "category": "electronics",
        "price": 49.99,
        "quantity": 2,
        "country": "US",
        "device_type": "desktop",
        "referrer": "google.com",
    }
    base.update(overrides)
    return base


class TestQualityGateIntegration:
    def test_clean_record_passes_all_rules(self):
        df = pd.DataFrame([_make_good_row()])
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        assert len(clean) == 1
        assert len(quarantine) == 0
        assert report.passed == 1
        assert report.failed == 0

    def test_null_user_id_quarantined(self):
        df = pd.DataFrame([_make_good_row(user_id=None)])
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        assert len(clean) == 0
        assert len(quarantine) == 1
        assert "_quarantine_reasons" in quarantine.columns

    def test_negative_price_quarantined(self):
        df = pd.DataFrame([_make_good_row(price=-10.0)])
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        assert len(quarantine) == 1
        reasons = quarantine["_quarantine_reasons"].iloc[0]
        assert "range_check" in reasons

    def test_future_timestamp_quarantined(self):
        future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        df = pd.DataFrame([_make_good_row(timestamp=future)])
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        assert len(quarantine) == 1

    def test_duplicate_event_ids(self):
        rows = [
            _make_good_row(event_id="dup_evt"),
            _make_good_row(event_id="dup_evt"),
        ]
        df = pd.DataFrame(rows)
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        # First occurrence passes uniqueness check; second fails
        assert len(quarantine) >= 1

    def test_invalid_event_type_quarantined(self):
        df = pd.DataFrame([_make_good_row(event_type="click")])
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        assert len(quarantine) == 1

    def test_report_totals_are_consistent(self):
        rows = [_make_good_row(event_id=f"e{i}") for i in range(5)]
        rows.append(_make_good_row(price=-1.0, event_id="bad_1"))  # fails range
        rows.append(_make_good_row(user_id=None, event_id="bad_2"))  # fails non-null
        df = pd.DataFrame(rows)
        gate = QualityGate()
        clean, quarantine, report = gate.validate(df)
        assert report.total_records == 7
        assert report.passed + report.failed == 7

    def test_empty_dataframe(self):
        gate = QualityGate()
        clean, quarantine, report = gate.validate(pd.DataFrame())
        assert report.total_records == 0
        assert report.passed == 0
        assert report.failed == 0

    def test_failure_breakdown_has_all_rules(self):
        df = pd.DataFrame([_make_good_row()])
        gate = QualityGate(rules=default_rules())
        _, _, report = gate.validate(df)
        rule_names = [r.rule_name for r in report.failure_breakdown_by_rule]
        assert "non_null_check" in rule_names
        assert "range_check" in rule_names
        assert "uniqueness_check" in rule_names

    def test_report_failure_rate_calculation(self):
        rows = [_make_good_row(event_id=f"e{i}") for i in range(8)]
        rows.append(_make_good_row(price=-1.0, event_id="b1"))
        rows.append(_make_good_row(user_id=None, event_id="b2"))
        df = pd.DataFrame(rows)
        gate = QualityGate()
        _, _, report = gate.validate(df)
        expected_rate = report.failed / report.total_records
        assert abs(report.failure_rate - expected_rate) < 1e-9

    def test_quarantine_reasons_column_populated(self):
        df = pd.DataFrame([
            _make_good_row(price=-5.0, event_id="bad_price"),
            _make_good_row(event_id="good"),
        ])
        gate = QualityGate()
        _, quarantine, _ = gate.validate(df)
        assert len(quarantine) == 1
        reason = quarantine["_quarantine_reasons"].iloc[0]
        assert len(reason) > 0
