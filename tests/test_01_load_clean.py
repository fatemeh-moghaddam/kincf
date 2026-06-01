"""Tests for scripts/01_load_clean.py.

test_fetch_kinase_metadata_known_ids  — live KLIFS API call, requires network.
test_output_file_exists_and_valid     — requires data/processed/measurements.csv.
test_no_duplicate_pairs               — requires data/processed/measurements.csv.

Run the pipeline first:  python scripts/01_load_clean.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Module name starts with a digit, so standard import won't work.
_spec = importlib.util.spec_from_file_location(
    "load_clean",
    Path(__file__).resolve().parents[1] / "scripts" / "01_load_clean.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fetch_kinase_metadata = _mod.fetch_kinase_metadata

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
MEASUREMENTS_CSV = DATA_ROOT / "clean" / "measurements.csv"
EXPECTED_COLUMNS = ["ligand_id", "kinase_id", "kinase_name", "kinase_family", "activity", "smiles"]


def test_fetch_kinase_metadata_known_ids():
    known = {"P00533": "EGFR", "P06239": "LCK", "P27361": "ERK1"}
    result = fetch_kinase_metadata(uniprot_ids=list(known.keys()))

    assert {"kinase_id", "kinase_name", "kinase_family"}.issubset(result.columns)
    assert set(result["kinase_id"]) == set(known.keys()), (
        f"Missing IDs: {set(known.keys()) - set(result['kinase_id'])}"
    )
    assert result[["kinase_name", "kinase_family"]].notna().all().all(), (
        "Null kinase_name or kinase_family for known IDs"
    )
    egfr_name = result.loc[result["kinase_id"] == "P00533", "kinase_name"].iloc[0]
    assert egfr_name == "EGFR", f"Expected EGFR for P00533, got {egfr_name!r}"


def test_output_file_exists_and_valid():
    if not MEASUREMENTS_CSV.exists():
        pytest.skip(f"{MEASUREMENTS_CSV} not found — run scripts/01_load_clean.py first")

    df = pd.read_csv(MEASUREMENTS_CSV)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert pd.api.types.is_float_dtype(df["activity"]), "activity column is not float"

    out_of_range = ~df["activity"].between(3, 12)
    assert not out_of_range.any(), (
        f"{out_of_range.sum()} activity values outside [3, 12]:\n"
        f"{df.loc[out_of_range, 'activity'].describe()}"
    )

    null_rate = df["kinase_name"].isna().mean()
    assert null_rate < 0.05, f"kinase_name null rate {null_rate:.2%} exceeds 5%"

    n_dups = df.duplicated(subset=["ligand_id", "kinase_id"]).sum()
    assert n_dups == 0, f"{n_dups} duplicate (ligand_id, kinase_id) pairs found"


def test_no_duplicate_pairs():
    if not MEASUREMENTS_CSV.exists():
        pytest.skip(f"{MEASUREMENTS_CSV} not found — run scripts/01_load_clean.py first")

    df = pd.read_csv(MEASUREMENTS_CSV)
    n_dups = df.duplicated(subset=["ligand_id", "kinase_id"]).sum()
    assert n_dups == 0, f"{n_dups} duplicate (ligand_id, kinase_id) pairs found"
