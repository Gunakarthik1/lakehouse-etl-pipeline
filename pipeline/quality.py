"""
Data Quality Gates — configurable validation rules with quarantine support.

Rules:
  - NonNullCheck       : critical columns must not be null
  - RangeCheck         : numeric column must be within [min_val, max_val]
  - TypeCheck          : column values must be castable to expected Python type
  - DateRangeCheck     : timestamp column must fall within [min_date, max_date]
  - UniquenessCheck    : column must contain no duplicates
  - CategoricalCheck   : column values must belong to a set of valid categories

The QualityGate.validate() method returns clean records, quarantined records,
and a QualityReport summarising all failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data structure
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    rule_name: str
    column: str
    passed: int
    failed: int
    failure_rate: float
    details: str = ""


@dataclass
class QualityReport:
    total_records: int
    passed: int
    failed: int
    failure_rate: float
    failure_breakdown_by_rule: list[RuleResult] = field(default_factory=list)
    quarantine_reasons: dict[int, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "passed": self.passed,
            "failed": self.failed,
            "failure_rate": round(self.failure_rate, 4),
            "failure_breakdown_by_rule": [
                {
                    "rule_name": r.rule_name,
                    "column": r.column,
                    "passed": r.passed,
                    "failed": r.failed,
                    "failure_rate": round(r.failure_rate, 4),
                    "details": r.details,
                }
                for r in self.failure_breakdown_by_rule
            ],
        }


# ---------------------------------------------------------------------------
# Individual rule implementations
# ---------------------------------------------------------------------------

class _QualityRule:
    """Base class for all quality rules."""

    name: str = "base_rule"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = record PASSES the rule."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NonNullCheck(_QualityRule):
    name = "non_null_check"

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.Series:
        present_cols = [c for c in self.columns if c in df.columns]
        if not present_cols:
            return pd.Series([True] * len(df), index=df.index)
        return df[present_cols].notnull().all(axis=1)

    def describe(self) -> str:
        return f"non_null_check(cols={self.columns})"


class RangeCheck(_QualityRule):
    name = "range_check"

    def __init__(self, column: str, min_val: float, max_val: float) -> None:
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series([True] * len(df), index=df.index)

        numeric = pd.to_numeric(df[self.column], errors="coerce")
        in_range = numeric.between(self.min_val, self.max_val)
        # Rows where numeric is NaN (non-numeric strings) also fail
        return in_range.fillna(False)

    def describe(self) -> str:
        return f"range_check(col={self.column}, min={self.min_val}, max={self.max_val})"


class TypeCheck(_QualityRule):
    name = "type_check"

    _TYPE_MAP = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }

    def __init__(self, column: str, expected_type: str) -> None:
        self.column = column
        self.expected_type = expected_type
        self._py_type = self._TYPE_MAP.get(expected_type)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns or self._py_type is None:
            return pd.Series([True] * len(df), index=df.index)

        def _castable(v: Any) -> bool:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return False
            try:
                self._py_type(v)
                return True
            except (ValueError, TypeError):
                return False

        return df[self.column].apply(_castable)

    def describe(self) -> str:
        return f"type_check(col={self.column}, type={self.expected_type})"


class DateRangeCheck(_QualityRule):
    name = "date_range_check"

    def __init__(
        self,
        column: str,
        min_date: datetime | None = None,
        max_date: datetime | None = None,
    ) -> None:
        self.column = column
        self.min_date = min_date
        self.max_date = max_date

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series([True] * len(df), index=df.index)

        parsed = pd.to_datetime(df[self.column], utc=True, errors="coerce")
        result = pd.Series([True] * len(df), index=df.index)

        # NaT rows (unparseable) always fail
        nat_mask = parsed.isna()
        result[nat_mask] = False

        if self.min_date is not None:
            min_ts = pd.Timestamp(self.min_date).tz_localize("UTC") if self.min_date.tzinfo is None else pd.Timestamp(self.min_date)
            result &= (parsed >= min_ts) | nat_mask
            result[nat_mask] = False

        if self.max_date is not None:
            max_ts = pd.Timestamp(self.max_date).tz_localize("UTC") if self.max_date.tzinfo is None else pd.Timestamp(self.max_date)
            result &= (parsed <= max_ts) | nat_mask
            result[nat_mask] = False

        return result

    def describe(self) -> str:
        return f"date_range_check(col={self.column}, min={self.min_date}, max={self.max_date})"


class UniquenessCheck(_QualityRule):
    name = "uniqueness_check"

    def __init__(self, column: str) -> None:
        self.column = column

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series([True] * len(df), index=df.index)
        return ~df[self.column].duplicated(keep="first")

    def describe(self) -> str:
        return f"uniqueness_check(col={self.column})"


class CategoricalCheck(_QualityRule):
    name = "categorical_check"

    def __init__(self, column: str, valid_values: list[Any]) -> None:
        self.column = column
        self.valid_values = set(valid_values)

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series([True] * len(df), index=df.index)
        return df[self.column].isin(self.valid_values)

    def describe(self) -> str:
        return f"categorical_check(col={self.column}, n_valid={len(self.valid_values)})"


# ---------------------------------------------------------------------------
# Quality Gate — orchestrates all rules
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = ["page_view", "add_to_cart", "purchase", "search", "review"]
VALID_DEVICE_TYPES = ["desktop", "mobile", "tablet"]
VALID_COUNTRIES = [
    "US", "GB", "DE", "FR", "CA", "AU", "IN", "BR", "JP", "MX",
    "ES", "IT", "NL", "SE", "NO", "DK", "PL", "RU", "CN", "KR",
    "SG", "HK", "TW", "ZA", "NG", "EG", "AR", "CL", "CO", "PE",
]


def default_rules() -> list[_QualityRule]:
    """
    Return the default set of quality rules for the e-commerce event schema.
    """
    now_utc = datetime.now(timezone.utc)
    min_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    return [
        NonNullCheck(["event_id", "user_id", "event_type", "timestamp"]),
        RangeCheck("price", min_val=0.01, max_val=100_000.0),
        RangeCheck("quantity", min_val=1, max_val=10_000),
        TypeCheck("price", "float"),
        TypeCheck("quantity", "int"),
        DateRangeCheck("timestamp", min_date=min_date, max_date=now_utc),
        UniquenessCheck("event_id"),
        CategoricalCheck("event_type", VALID_EVENT_TYPES),
        CategoricalCheck("device_type", VALID_DEVICE_TYPES),
        CategoricalCheck("country", VALID_COUNTRIES),
    ]


class QualityGate:
    """
    Orchestrates a sequence of quality rules against a DataFrame.

    Usage::

        gate = QualityGate()  # uses default_rules()
        clean_df, quarantine_df, report = gate.validate(df)
    """

    def __init__(self, rules: list[_QualityRule] | None = None) -> None:
        self.rules: list[_QualityRule] = rules if rules is not None else default_rules()

    def add_rule(self, rule: _QualityRule) -> None:
        self.rules.append(rule)

    def validate(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
        """
        Apply all quality rules to df.

        Returns:
            clean_df       : Records that passed ALL rules.
            quarantine_df  : Records that failed at least one rule,
                             with a '_quarantine_reasons' column appended.
            report         : QualityReport with per-rule breakdowns.
        """
        if df.empty:
            empty_report = QualityReport(
                total_records=0, passed=0, failed=0, failure_rate=0.0
            )
            return df.copy(), df.copy(), empty_report

        total = len(df)
        # Track which rows fail, and why
        row_failures: dict[int, list[str]] = {idx: [] for idx in df.index}
        rule_results: list[RuleResult] = []

        for rule in self.rules:
            passed_mask: pd.Series = rule.apply(df)

            failed_count = int((~passed_mask).sum())
            passed_count = total - failed_count
            failure_rate = failed_count / total if total > 0 else 0.0

            rule_results.append(
                RuleResult(
                    rule_name=rule.name,
                    column=getattr(rule, "column", getattr(rule, "columns", "")),
                    passed=passed_count,
                    failed=failed_count,
                    failure_rate=failure_rate,
                    details=rule.describe(),
                )
            )

            for idx in df.index[~passed_mask]:
                row_failures[idx].append(rule.describe())

        # Determine clean vs quarantine
        failed_indices = [idx for idx, reasons in row_failures.items() if reasons]
        passed_indices = [idx for idx, reasons in row_failures.items() if not reasons]

        clean_df = df.loc[passed_indices].copy()
        quarantine_df = df.loc[failed_indices].copy()

        if not quarantine_df.empty:
            quarantine_df["_quarantine_reasons"] = [
                "; ".join(row_failures[idx]) for idx in failed_indices
            ]

        n_failed = len(failed_indices)
        n_passed = len(passed_indices)

        report = QualityReport(
            total_records=total,
            passed=n_passed,
            failed=n_failed,
            failure_rate=n_failed / total if total > 0 else 0.0,
            failure_breakdown_by_rule=rule_results,
            quarantine_reasons={
                idx: reasons
                for idx, reasons in row_failures.items()
                if reasons
            },
        )

        logger.info(
            "Quality validation — total=%d passed=%d failed=%d rate=%.2f%%",
            total,
            n_passed,
            n_failed,
            report.failure_rate * 100,
        )

        return clean_df, quarantine_df, report
