"""
Realistic synthetic e-commerce event data generator.
Produces events with configurable corruption for pipeline testing.
"""

import random
import uuid
import copy
from datetime import datetime, timedelta, timezone
from typing import Any

PRODUCT_CATALOG = [
    ("prod_001", "Wireless Headphones", "Electronics", 89.99),
    ("prod_002", "Running Shoes", "Footwear", 124.50),
    ("prod_003", "Coffee Maker", "Appliances", 59.99),
    ("prod_004", "Yoga Mat", "Sports", 34.99),
    ("prod_005", "Laptop Stand", "Electronics", 49.99),
    ("prod_006", "Water Bottle", "Sports", 22.00),
    ("prod_007", "Desk Lamp", "Home", 38.75),
    ("prod_008", "Mechanical Keyboard", "Electronics", 149.00),
    ("prod_009", "Backpack", "Bags", 79.99),
    ("prod_010", "Sunglasses", "Accessories", 55.00),
    ("prod_011", "Protein Powder", "Health", 44.99),
    ("prod_012", "Smart Watch", "Electronics", 199.99),
    ("prod_013", "Kitchen Knife Set", "Appliances", 89.00),
    ("prod_014", "Resistance Bands", "Sports", 18.50),
    ("prod_015", "Notebook (Paper)", "Stationery", 9.99),
]

EVENT_TYPES = ["page_view", "add_to_cart", "purchase", "search", "review"]
COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "IN", "BR", "JP", "MX"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
REFERRERS = [
    "google.com", "facebook.com", "instagram.com", "email_campaign",
    "direct", "twitter.com", "youtube.com", "affiliate_partner",
]


def _random_timestamp(
    start_days_ago: int = 90,
    end_days_ago: int = 0,
) -> str:
    now = datetime.now(timezone.utc)
    delta_seconds = random.randint(
        end_days_ago * 86400,
        start_days_ago * 86400,
    )
    ts = now - timedelta(seconds=delta_seconds)
    return ts.isoformat()


def _make_clean_event() -> dict[str, Any]:
    product = random.choice(PRODUCT_CATALOG)
    event_type = random.choice(EVENT_TYPES)
    quantity = 1 if event_type != "purchase" else random.randint(1, 5)
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": f"user_{random.randint(1000, 9999)}",
        "session_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": _random_timestamp(),
        "product_id": product[0],
        "product_name": product[1],
        "category": product[2],
        "price": round(product[3], 2),
        "quantity": quantity,
        "country": random.choice(COUNTRIES),
        "device_type": random.choice(DEVICE_TYPES),
        "referrer": random.choice(REFERRERS),
    }


CORRUPTION_STRATEGIES = [
    # Null out a critical field
    lambda e: {**e, "user_id": None},
    lambda e: {**e, "event_type": None},
    lambda e: {**e, "timestamp": None},
    # Wrong type for price
    lambda e: {**e, "price": "not_a_number"},
    # Negative price
    lambda e: {**e, "price": round(random.uniform(-500, -0.01), 2)},
    # Future timestamp (30–365 days ahead)
    lambda e: {
        **e,
        "timestamp": (
            datetime.now(timezone.utc) + timedelta(days=random.randint(30, 365))
        ).isoformat(),
    },
    # Invalid event type
    lambda e: {**e, "event_type": random.choice(["click", "impression", "UNKNOWN", ""])},
    # Invalid country code
    lambda e: {**e, "country": random.choice(["XX", "ZZ", "1234", "United States"])},
    # Quantity zero or negative
    lambda e: {**e, "quantity": random.randint(-10, 0)},
    # Null price
    lambda e: {**e, "price": None},
]


def generate_batch(
    n: int = 1000,
    corruption_rate: float = 0.08,
) -> list[dict[str, Any]]:
    """
    Generate n synthetic e-commerce events.

    Args:
        n: Number of events to generate.
        corruption_rate: Fraction of records that will contain deliberate
            data quality issues (nulls, wrong types, future timestamps,
            negative prices, duplicate event_ids).

    Returns:
        List of event dicts, some intentionally corrupted.
    """
    records: list[dict[str, Any]] = []
    n_corrupt = max(1, int(n * corruption_rate))
    corrupt_indices: set[int] = set(random.sample(range(n), min(n_corrupt, n)))

    # Reserve a few indices for duplicate event_ids
    n_dupes = max(1, int(n * 0.02))
    dupe_pool: list[str] = []

    for i in range(n):
        event = _make_clean_event()

        if i in corrupt_indices:
            strategy = random.choice(CORRUPTION_STRATEGIES)
            event = strategy(copy.deepcopy(event))

        records.append(event)

        # Collect some event_ids to duplicate later
        if len(dupe_pool) < 20 and isinstance(event.get("event_id"), str):
            dupe_pool.append(event["event_id"])

    # Inject duplicate event_ids into random positions
    if dupe_pool:
        for _ in range(n_dupes):
            target_idx = random.randint(0, n - 1)
            records[target_idx] = {
                **records[target_idx],
                "event_id": random.choice(dupe_pool),
            }

    random.shuffle(records)
    return records
