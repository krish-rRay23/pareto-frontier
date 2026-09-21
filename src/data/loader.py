"""Data loading and synthetic catalog dataset generation."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_dataset(
    path: Path,
    expected_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Load CSV or Parquet dataset safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in [".csv", ".tsv"]:
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"File {path.name} missing expected columns: {missing}")

    return df


def generate_synthetic_catalog_data(
    n_samples: int = 500,
    seed: int = 42,
    is_test: bool = False,
) -> pd.DataFrame:
    """Generate realistic synthetic Amazon catalog data for local pipeline verification."""
    rng = np.random.default_rng(seed)

    sample_ids = [f"AMZN_{i:06d}" for i in range(1, n_samples + 1)]
    brands = [
        "Samsung",
        "Philips",
        "Dabur",
        "Tata",
        "Sony",
        "Logitech",
        "Nestle",
        "Boat",
        "Milton",
        "Puma",
        "Generic Brand",
    ]
    units = ["g", "kg", "ml", "L", "count", "oz", "pack"]
    categories = ["Electronics", "Grocery", "Personal Care", "Home & Kitchen", "Apparel"]

    rows = []
    for i in range(n_samples):
        brand = rng.choice(brands)
        cat = rng.choice(categories)
        unit = rng.choice(units)
        val = rng.choice([50, 100, 250, 500, 750, 1000, 2, 4, 10])
        pack = rng.choice([1, 1, 1, 2, 3, 4, 6, 10])

        pack_str = f"Pack of {pack}" if pack > 1 else "Single Unit"
        title = f"{brand} {cat} Item - {val}{unit} ({pack_str})"
        desc = (
            f"High quality original {brand} product for {cat}. "
            f"Net quantity: {val} {unit}. Standard warranty included. Genuine retail pack."
        )
        bullet_points = (
            f"Brand: {brand} | Quantity: {val}{unit} | Type: {pack_str} | Certified item."
        )

        # Synthetic price generation loosely correlated with quantity, pack, and brand
        base_price = 150.0 + (val * 0.4) * pack + (len(brand) * 20.0)
        noise = rng.lognormal(mean=0.0, sigma=0.25)
        price = round(float(base_price * noise), 2)

        row = {
            "sample_id": sample_ids[i],
            "catalog_title": title,
            "description": desc,
            "bullet_points": bullet_points,
            "category": cat,
        }
        if not is_test:
            row["target_price"] = price

        rows.append(row)

    return pd.DataFrame(rows)
