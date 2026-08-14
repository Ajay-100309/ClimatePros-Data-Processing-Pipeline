"""Configuration for the new-dispatch pipeline.

All secrets come from the untracked .env in the working directory. Values are
stripped (the gateway BASE_URL was observed with a trailing space). Validation
reports missing key NAMES only — never values.
"""
import os
import sys
import hashlib

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(HERE, ".env"))


def _env(name, required=True):
    v = os.getenv(name)
    if v is not None:
        v = v.strip()
    if required and not v:
        return None
    return v


_REQUIRED = ["API_KEY", "BASE_URL", "MODEL",
             "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
_missing = [k for k in _REQUIRED if not _env(k)]
if _missing:
    sys.exit(f"Missing required .env keys: {', '.join(_missing)}")

API_KEY = _env("API_KEY")
BASE_URL = _env("BASE_URL")
MODEL = _env("MODEL")

DB = dict(
    server=_env("DB_HOST"),
    port=int(_env("DB_PORT")),
    database=_env("DB_NAME"),
    user=_env("DB_USER"),
    password=_env("DB_PASSWORD"),
)

EMBED_MODEL = "nomic-embed"
EMBED_PREFIX = ""
EMBED_BATCH = 100
EMBED_TRUNCATE = 4000

N_CANDIDATES = 5
SIM_FLOOR = 0.60
NOTE_CHUNK = 20
MAX_WORKERS = 1
CHECKPOINT_EVERY = 10

# DB snapshot goes quiet after early June 2026; the proven sample cutoff
RECEIVED_MIN = "2013-01-01"
RECEIVED_CUTOFF = "2026-07-01"
MIN_NOTE_LEN = 40

STATE_DIR = os.path.join(HERE, "state")
OUTPUT_DIR = os.path.join(HERE, "output")
BATCH_ARCHIVE_DIR = os.path.join(STATE_DIR, "batches")

LEDGER_FILE = os.path.join(STATE_DIR, "ledger.json")
CASEMAP_FILE = os.path.join(STATE_DIR, "casemap.json")
NOTES_CLASS_FILE = os.path.join(STATE_DIR, "notes_class.json")
EXTRACT_FILE = os.path.join(STATE_DIR, "extract.json")
DISPATCH_META_FILE = os.path.join(STATE_DIR, "dispatch_meta.json")
BATCH_FILE = os.path.join(STATE_DIR, "batch_current.json")
EMB_NPY = os.path.join(STATE_DIR, "embeddings.npy")
EMB_INDEX = os.path.join(STATE_DIR, "embeddings_index.json")

CHROMA_DIR = os.path.join(OUTPUT_DIR, "chroma_cases_extracted")
CHROMA_COLLECTION = "root_cause_cases"

OUT_MAPPED = os.path.join(OUTPUT_DIR, "Dispatch_CaseMapped.xlsx")
OUT_CASES = os.path.join(OUTPUT_DIR, "case_summary_extracted.xlsx")
OUT_GROWTH_XLSX = os.path.join(OUTPUT_DIR, "case_growth.xlsx")
OUT_GROWTH_PNG = os.path.join(OUTPUT_DIR, "case_growth.png")

PROMPT_NOTES = os.path.join(HERE, "prompt_notes.txt")
PROMPT_EXTRACT = os.path.join(HERE, "prompt_extract.txt")
PROMPT_CASEMAP = os.path.join(HERE, "prompt_casemap.txt")

NO_FAULT_PLACEHOLDER = "No technical fault identified in dispatch notes."

XLSX_CELL_LIMIT = 32000


def read_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prompt_sha(path):
    return hashlib.sha256(read_prompt(path).encode()).hexdigest()


def casemap_fingerprint():
    return {
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "embed_prefix": EMBED_PREFIX,
        "sim_floor": SIM_FLOOR,
        "n_candidates": N_CANDIDATES,
        "prompt_sha256": prompt_sha(PROMPT_CASEMAP),
    }


def norm_guid(value):
    return str(value).strip().upper()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
