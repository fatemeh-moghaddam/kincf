"""Load and clean the raw kinodata SDF into a flat measurements table.

Reads  : data/raw/kinodata_docked_v2.sdf.gz
Fetches: kinase gene names and families from the KLIFS REST API
         (cached to data/raw/klifs_kinase_info.csv on first run)
Writes : data/clean/measurements.csv

Output columns:
  ligand_id     – molecule ChEMBL ID   (molecule_dictionary.chembl_id)
  kinase_id     – kinase UniProt ID    (UniprotID)
  kinase_name   – HGNC gene symbol     (from KLIFS API, e.g. EGFR)
  kinase_family – KLIFS kinase family  (from KLIFS API, e.g. EGFR)
  activity      – pIC50 value          (activities.standard_value, pIC50 rows only)
  smiles        – canonical SMILES     (compound_structures.canonical_smiles)

Duplicate (ligand_id, kinase_id) pairs are resolved by keeping the measurement
closest to the per-pair median activity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from rdkit.Chem import PandasTools

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SDF_PATH = DATA_ROOT / "raw" / "kinodata_docked_v2.sdf.gz"
KLIFS_KINASE_URL = "https://dev.klifs.net/api_v2/kinases?species=Human"


def load_raw_sdf(sdf_path: Path) -> pd.DataFrame:
    """Load SDF with RDKit PandasTools, return raw dataframe."""
    return PandasTools.LoadSDF(
        str(sdf_path),
        smilesName="compound_structures.canonical_smiles",
        molColName="molecule",
        embedProps=True,
        removeHs=True,
    )


def fetch_kinase_metadata(
    uniprot_ids: list[str] | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch kinase gene names and families from the KLIFS REST API.

    Returns a DataFrame with columns: kinase_id (UniProt), kinase_name, kinase_family.
    If uniprot_ids is given, the result is filtered to those IDs.
    If cache_path is given, the full table is written on first call and read on subsequent ones.
    """
    if cache_path is not None and cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        print(f"Fetching kinase info from {KLIFS_KINASE_URL} ...")
        response = requests.get(KLIFS_KINASE_URL, timeout=30)
        response.raise_for_status()
        df = (
            pd.DataFrame(response.json())[["UniProt", "name", "family"]]
            .rename(columns={"UniProt": "kinase_id", "name": "kinase_name", "family": "kinase_family"})
            .drop_duplicates(subset="kinase_id")
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"Cached kinase info ({len(df)} kinases) → {cache_path}")

    if uniprot_ids is not None:
        df = df[df["kinase_id"].isin(uniprot_ids)].reset_index(drop=True)

    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Keep only pIC50 activity type
    - Select and rename relevant columns to: ligand_id, kinase_id, activity, smiles
    - Cast activity to float, drop nulls
    - Resolve duplicate (ligand_id, kinase_id) pairs by keeping the row
      closest to the median activity for that pair
    """
    print("Columns found:", df.columns.tolist())
    print("dtypes:\n", df.dtypes.to_string())

    # Confirmed column names from kinodata_docked_v2.sdf.gz:
    #   ligand ChEMBL ID   → molecule_dictionary.chembl_id
    #   kinase UniProt ID  → UniprotID
    #   affinity value     → activities.standard_value
    #   affinity type      → activities.standard_type
    #   canonical SMILES   → compound_structures.canonical_smiles
    col_activity_type = "activities.standard_type"
    col_activity_value = "activities.standard_value"
    col_ligand_id = "molecule_dictionary.chembl_id"
    col_kinase_id = "UniprotID"
    col_smiles = "compound_structures.canonical_smiles"

    df = df[df[col_activity_type] == "pIC50"].copy()

    df = df[[col_ligand_id, col_kinase_id, col_activity_value, col_smiles]].rename(
        columns={
            col_ligand_id: "ligand_id",
            col_kinase_id: "kinase_id",
            col_activity_value: "activity",
            col_smiles: "smiles",
        }
    )

    df["activity"] = pd.to_numeric(df["activity"], errors="coerce")
    df = df.dropna(subset=["activity"])

    # Resolve duplicates: keep the row closest to the per-pair median activity
    medians = df.groupby(["ligand_id", "kinase_id"])["activity"].transform("median")
    df = df.assign(_dist=(df["activity"] - medians).abs())
    df = (
        df.sort_values("_dist")
        .drop_duplicates(subset=["ligand_id", "kinase_id"], keep="first")
        .drop(columns="_dist")
        .reset_index(drop=True)
    )

    return df


def report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    """Print:
    - Row count before and after cleaning
    - Null rate in activity before drop
    - Unique ligand count
    - Unique kinase count
    - activity min / mean / max
    - Full list of unique (kinase_id, kinase_name) pairs
    """
    pic50_mask = raw_df["activities.standard_type"] == "pIC50"
    activity_raw = pd.to_numeric(
        raw_df.loc[pic50_mask, "activities.standard_value"], errors="coerce"
    )
    null_rate = activity_raw.isna().mean()

    print(f"\n{'Rows (raw):':<26} {len(raw_df):>10,}")
    print(f"{'Rows (pIC50 subset):':<26} {pic50_mask.sum():>10,}")
    print(f"{'Rows (clean):':<26} {len(clean_df):>10,}")
    print(f"{'activity null rate:':<26} {null_rate:>10.2%}")
    print(f"{'Unique ligands:':<26} {clean_df['ligand_id'].nunique():>10,}")
    print(f"{'Unique kinases:':<26} {clean_df['kinase_id'].nunique():>10,}")
    print(f"{'activity min:':<26} {clean_df['activity'].min():>10.3f}")
    print(f"{'activity mean:':<26} {clean_df['activity'].mean():>10.3f}")
    print(f"{'activity max:':<26} {clean_df['activity'].max():>10.3f}")

    kinase_cols = [c for c in ["kinase_id", "kinase_name", "kinase_family"] if c in clean_df.columns]
    kinase_pairs = clean_df[kinase_cols].drop_duplicates().sort_values("kinase_id")
    print(f"\nUnique kinases ({len(kinase_pairs)}):")
    print(kinase_pairs.to_string(index=False))


def main() -> None:
    out_path = DATA_ROOT / "clean" / "measurements.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SDF from {SDF_PATH} ...")
    raw_df = load_raw_sdf(SDF_PATH)
    print(f"Loaded {len(raw_df):,} rows.")

    clean_df = filter_and_clean(raw_df)

    kinase_info = fetch_kinase_metadata(cache_path=DATA_ROOT / "raw" / "klifs_kinase_info.csv")
    clean_df = clean_df.merge(kinase_info, on="kinase_id", how="left")
    clean_df = clean_df[["ligand_id", "kinase_id", "kinase_name", "kinase_family", "activity", "smiles"]]

    report(raw_df, clean_df)

    clean_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(clean_df):,} rows → {out_path}")


if __name__ == "__main__":
    main()
