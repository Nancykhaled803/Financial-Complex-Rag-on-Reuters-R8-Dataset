import re
from pathlib import Path

import pandas as pd

from rag.config import DATA_PATH, ON_KAGGLE, ROOT

REQUIRED_COLUMNS = ("label", "text", "source")
CLEAN_FILENAMES = ("reuters_clean.parquet", "reuters_clean.csv")
RAW_TRAIN_NAME = "r8-train-all-terms.txt"
RAW_TEST_NAME = "r8-test-all-terms.txt"

ABBREVIATIONS = {
    "mln": "million",
    "dlrs": "dollars",
    "dlr": "dollar",
    "cts": "cents",
    "pct": "percent",
    "stg": "sterling",
    "bpd": "barrels per day",
    "bbl": "barrels",
    "bbls": "barrels",
    "shr": "share",
    "shrs": "shares",
    "qtr": "quarter",
    "qtrs": "quarters",
    "qtly": "quarterly",
    "yr": "year",
    "mths": "months",
    "revs": "revenues",
    "avg": "average",
    "div": "dividend",
    "oper": "operating",
    "vs": "versus",
    "th qtr": "third quarter",
}
_ABBR_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, ABBREVIATIONS), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def _search_roots() -> list[Path]:
    roots = [ROOT, ROOT / "data"]
    if ON_KAGGLE:
        roots.append(Path("/kaggle/input"))
    return roots


def _find_file(name: str) -> Path | None:
    for root in _search_roots():
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        matches = [path for path in root.rglob(name) if path.is_file()]
        if matches:
            return matches[0]
    return None


def _resolve_clean_path(path: str | None) -> Path | None:
    if path:
        given = Path(path)
        return given if given.is_file() else None
    if DATA_PATH and Path(DATA_PATH).is_file():
        return Path(DATA_PATH)
    for name in CLEAN_FILENAMES:
        found = _find_file(name)
        if found is not None:
            return found
    return None


def _extract_source_and_truncate(text: str) -> tuple[str, str]:
    words = str(text).strip().split()
    if words and words[-1].lower() in {"reuter", "reuters"}:
        return " ".join(words[:-1]), "reuters"
    return str(text).strip(), "unknown"


def _expand_abbreviations(text: str) -> str:
    return _ABBR_PATTERN.sub(lambda match: ABBREVIATIONS[match.group(0).lower()], str(text))


def _load_raw_reuters() -> pd.DataFrame:
    train_path = _find_file(RAW_TRAIN_NAME)
    test_path = _find_file(RAW_TEST_NAME)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Clean data was not found, and the raw R8 files are missing too. "
            "Add reuters_clean.csv, or attach r8-train-all-terms.txt and r8-test-all-terms.txt."
        )

    # print("Clean file not found. Building it from:")
    # print(" ", train_path)
    # print(" ", test_path)

    df_train = pd.read_csv(train_path, sep="\t", names=["label", "text"])
    df_test = pd.read_csv(test_path, sep="\t", names=["label", "text"])
    df_train["split"] = "train"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)

    processed = df["text"].apply(_extract_source_and_truncate)
    df["text"] = processed.str[0]
    df["source"] = processed.str[1]
    df["original_text"] = df["text"]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    df["text"] = df["text"].apply(_expand_abbreviations)
    return df[["label", "text", "source", "split", "original_text"]]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Clean data is missing columns: {missing}")

    df = df.copy()
    if "split" not in df.columns:
        df["split"] = ""
    if "original_text" not in df.columns:
        df["original_text"] = df["text"]

    df["text"] = df["text"].fillna("").astype(str)
    df["original_text"] = df["original_text"].fillna("").astype(str)
    df["label"] = df["label"].astype(str)
    df["source"] = df["source"].astype(str)
    df["split"] = df["split"].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the cleaned Reuters table.

    Search order:
    1. The path argument, or DATA_PATH from .env
    2. reuters_clean.parquet / reuters_clean.csv next to the project or under /kaggle/input
    3. The raw r8 train and test files, which are cleaned in memory
    """
    clean_path = _resolve_clean_path(path)
    if clean_path is not None:
        # print("Using clean file:", clean_path)
        if clean_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(clean_path)
        else:
            df = pd.read_csv(clean_path)
        return _finalize(df)

    return _finalize(_load_raw_reuters())


def get_categories(df: pd.DataFrame) -> list[str]:
    """Sorted category names present in the loaded data."""
    return sorted(df["label"].dropna().astype(str).unique())
