"""
Taxonomy mapping module for ToriNet.

Maps between eBird 2024 species codes (project standard) and:
  - BirdNET V2.4 species (scientific names)
  - Google Perch v2 species (eBird 2021 codes / label indices)

Data sources:
  - species_list.csv: Master species list with eBird and BirdNET mappings
  - Perch label.csv: eBird 2021 codes used by Google Perch v2

Usage:
  from taxonomy_maps import build_taxonomy_maps
  maps = build_taxonomy_maps(["japrob2", "narfly2", "wbwsta1"])
  maps.species_to_birdnet_sciname["wbwsta1"]  # -> "Saxicola maurus"
  maps.species_to_perch_idx["narfly2"]        # -> 5872
  maps.classifier_coverage["japrob2"]         # -> "both"
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
SPECIES_LIST_PATH = Path(__file__).resolve().parent.parent / "01_species_list" / "species_list.csv"

# Perch v2 label.csv is an asset shipped with the TF Hub model bundle. The
# cache directory is resolvable at runtime via tensorflow_hub (falls back to
# $TFHUB_CACHE_DIR or /tmp/tfhub_modules), so we locate label.csv dynamically
# instead of hard-coding the hash path (which disappears on /tmp cleanup).
PERCH_MODEL_URL = "https://tfhub.dev/google/bird-vocalization-classifier/2"


def _resolve_perch_label_path() -> Path:
    """Return the path to Perch v2 label.csv, downloading the model if needed."""
    import tensorflow_hub as hub

    resolved_dir = Path(hub.resolve(PERCH_MODEL_URL))
    return resolved_dir / "assets" / "label.csv"

# ---------------------------------------------------------------------------
# eBird 2024 -> 2021 code mapping for Perch
# ---------------------------------------------------------------------------
# Between eBird/Clements v2021 and v2024, some species codes changed due to
# taxonomic splits, genus reassignments, or numbering changes. This dict maps
# the new 2024 code to the old 2021 code present in the Perch label file.
#
# Verified programmatically: for each entry, the 2021 code exists in Perch's
# label.csv and the 2024 code exists in species_list.csv but NOT in Perch.
EBIRD2024_TO_EBIRD2021: dict[str, str] = {
    # Code renumbering / minor revisions
    "gubter2": "gubter1",   # Gull-billed Tern (Gelochelidon nilotica)
    "integr1": "integr",    # Intermediate Egret (Ardea intermedia)
    "norgos1": "norgos",    # Northern Goshawk (Astur gentilis, was Accipiter)
    "easmah1": "easmah2",   # Eastern Marsh Harrier (Circus spilonotus)
    "chgshr1": "chgshr2",   # Chinese Gray Shrike (Lanius sphenocercus)
    "refblu1": "refblu",    # Red-flanked Bluetail (Tarsiger cyanurus)
    "rinphe1": "rinphe",    # Common Pheasant (Phasianus colchicus)
    # Taxonomic splits (2024 species -> pre-split 2021 code)
    "whwsco2": "whwsco",    # White-winged Scoter (Melanitta deglandi) -> pre-split
    "wehpit1": "hoopit1",   # Western Hooded Pitta (Pitta sordida) -> Hooded Pitta
    "eurnut3": "eurnut1",   # Eurasian Nutcracker (Nucifraga caryocatactes) -> pre-split
    "cintit13": "gretit4",  # Cinereous Tit (Parus cinereus, Japan) -> Great Tit group
    "pacswa5": "pacswa1",   # Pacific Swallow (Hirundo tahitica) -> pre-split
    "y00621": "rerswa1",    # Red-rumped Swallow (Cecropis daurica) -> old code
    "leswhi4": "leswhi1",   # Lesser Whitethroat (Curruca curruca) -> pre-split
    "japrob2": "japrob1",   # Japanese Robin (Larvivora akahige) -> pre-split
    "ryurob2": "ryurob1",   # Ryukyu Robin (Larvivora komadori) -> pre-split
    "ryurob3": "ryurob1",   # Okinawa Robin (Larvivora namiyei) -> was Ryukyu Robin
    "narfly2": "narfly1",   # Narcissus Flycatcher (Ficedula narcissina) -> pre-split
    "narfly3": "narfly1",   # Owston's Flycatcher (Ficedula owstoni) -> was Narcissus Fly.
    "redpol1": "comred",    # Common Redpoll (Acanthis flammea) -> old code
}

# Species with NO equivalent in Perch v2 (absent from eBird 2021 label set entirely)
PERCH_UNMAPPABLE: set[str] = {
    "gnwtea",   # Green-winged Teal (Anas crecca) - not in Perch 2021 labels
    "lessap2",  # Lesser Sand-Plover (Anarhynchus mongolus) - not in Perch 2021 labels
    "bkfbun1",  # Black-faced Bunting (Emberiza spodocephala) - not in Perch 2021 labels
    "bkfbun2",  # Masked Bunting (Emberiza personata) - not in Perch 2021 labels
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class TaxonomyMaps:
    """Container for taxonomy mapping results."""

    # eBird 2024 code -> BirdNET scientific name (for BirdNET lookup)
    species_to_birdnet_sciname: dict[str, str] = field(default_factory=dict)

    # eBird 2024 code -> Perch label index (row index in label.csv, 0-based)
    species_to_perch_idx: dict[str, int] = field(default_factory=dict)

    # eBird 2024 code -> coverage category
    #   "both"    : covered by BirdNET and Perch
    #   "birdnet" : covered by BirdNET only
    #   "perch"   : covered by Perch only
    #   "none"    : not covered by either classifier
    classifier_coverage: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------
def _load_species_list() -> list[dict[str, str]]:
    """Load species_list.csv and return rows as list of dicts."""
    rows = []
    with open(SPECIES_LIST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_perch_labels() -> dict[str, int]:
    """Load Perch label.csv and return {ebird2021_code: row_index}."""
    code_to_idx: dict[str, int] = {}
    with open(_resolve_perch_label_path(), newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            code_to_idx[row["ebird2021"]] = idx
    return code_to_idx


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_taxonomy_maps(species_codes: list[str]) -> TaxonomyMaps:
    """Build mapping from eBird 2024 codes to BirdNET and Perch indices.

    Args:
        species_codes: List of eBird 2024 species codes to map.

    Returns:
        TaxonomyMaps with BirdNET scientific names, Perch label indices,
        and per-species classifier coverage.
    """
    # Load reference data
    species_rows = _load_species_list()
    code_to_row = {row["ebird_species_code"]: row for row in species_rows}
    perch_code_to_idx = _load_perch_labels()

    maps = TaxonomyMaps()

    for code in species_codes:
        row = code_to_row.get(code)
        if row is None:
            maps.classifier_coverage[code] = "none"
            continue

        has_birdnet = False
        has_perch = False

        # --- BirdNET mapping ---
        if row["birdnet_matched"] == "True" and row["birdnet_sciname"]:
            maps.species_to_birdnet_sciname[code] = row["birdnet_sciname"]
            has_birdnet = True

        # --- Perch mapping ---
        # Try direct code match first, then 2024->2021 mapping
        perch_code = None
        if code in perch_code_to_idx:
            perch_code = code
        elif code in EBIRD2024_TO_EBIRD2021:
            old_code = EBIRD2024_TO_EBIRD2021[code]
            if old_code in perch_code_to_idx:
                perch_code = old_code

        if perch_code is not None:
            maps.species_to_perch_idx[code] = perch_code_to_idx[perch_code]
            has_perch = True

        # --- Coverage ---
        if has_birdnet and has_perch:
            maps.classifier_coverage[code] = "both"
        elif has_birdnet:
            maps.classifier_coverage[code] = "birdnet"
        elif has_perch:
            maps.classifier_coverage[code] = "perch"
        else:
            maps.classifier_coverage[code] = "none"

    return maps


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _print_coverage_stats() -> None:
    """Print coverage statistics for all 688 eBird-matched species."""
    species_rows = _load_species_list()
    all_codes = [
        row["ebird_species_code"]
        for row in species_rows
        if row["ebird_matched"] == "True"
    ]

    print(f"Building taxonomy maps for {len(all_codes)} species...")
    maps = build_taxonomy_maps(all_codes)

    # Count coverage categories
    counts: dict[str, int] = {"both": 0, "birdnet": 0, "perch": 0, "none": 0}
    for code in all_codes:
        cat = maps.classifier_coverage.get(code, "none")
        counts[cat] += 1

    total = len(all_codes)
    birdnet_total = counts["both"] + counts["birdnet"]
    perch_total = counts["both"] + counts["perch"]

    print(f"\n{'Category':<12s} {'Count':>6s} {'Pct':>7s}")
    print("-" * 27)
    print(f"{'Both':<12s} {counts['both']:>6d} {counts['both']/total*100:>6.1f}%")
    print(f"{'BirdNET only':<12s} {counts['birdnet']:>6d} {counts['birdnet']/total*100:>6.1f}%")
    print(f"{'Perch only':<12s} {counts['perch']:>6d} {counts['perch']/total*100:>6.1f}%")
    print(f"{'Neither':<12s} {counts['none']:>6d} {counts['none']/total*100:>6.1f}%")
    print("-" * 27)
    print(f"{'Total':<12s} {total:>6d}")
    print(f"\nBirdNET coverage: {birdnet_total}/{total} ({birdnet_total/total*100:.1f}%)")
    print(f"Perch coverage:   {perch_total}/{total} ({perch_total/total*100:.1f}%)")

    # Show species with no coverage
    none_species = [code for code in all_codes if maps.classifier_coverage.get(code) == "none"]
    if none_species:
        print(f"\nSpecies with no classifier coverage ({len(none_species)}):")
        code_to_row = {row["ebird_species_code"]: row for row in species_rows}
        for code in none_species:
            row = code_to_row[code]
            print(f"  {code:<12s} {row['scientific_name']:<40s} {row['japanese_name']}")

    # Show Perch-unmappable species
    unmappable_in_list = [code for code in all_codes if code in PERCH_UNMAPPABLE]
    if unmappable_in_list:
        print(f"\nPerch-unmappable species (no eBird 2021 equivalent, {len(unmappable_in_list)}):")
        code_to_row = {row["ebird_species_code"]: row for row in species_rows}
        for code in unmappable_in_list:
            row = code_to_row[code]
            print(f"  {code:<12s} {row['scientific_name']:<40s} {row['japanese_name']}")

    # Show 2024->2021 code mappings used
    mapped_codes = [code for code in all_codes if code in EBIRD2024_TO_EBIRD2021]
    if mapped_codes:
        print(f"\neBird 2024->2021 code mappings applied ({len(mapped_codes)}):")
        for code in mapped_codes:
            old = EBIRD2024_TO_EBIRD2021[code]
            print(f"  {code:<12s} -> {old}")


if __name__ == "__main__":
    _print_coverage_stats()
