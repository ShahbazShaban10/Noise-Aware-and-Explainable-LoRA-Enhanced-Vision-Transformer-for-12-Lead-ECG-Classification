"""SNOMED CT -> 7-class label resolution for Chapman-Shaoxing WFDB records.

Chapman records are multi-label (`#Dx: 164889003,59118001,164934002`). The study is a
single-label 7-class problem, so a record must be reduced to one class. That reduction is
the single most consequential preprocessing decision in the whole pipeline and it is
therefore data-driven (`environment/data/class_map_7.json`), auditable, and covered by an
exact-match test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CLASS_NAMES, set_active_class_names

_DX_RE = re.compile(r"^#\s*Dx\s*:(.*)$", re.IGNORECASE)
_CODE_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class LabelResolution:
    """The outcome of resolving one record's codes to one class."""

    record_id: str
    label: str
    label_index: int
    matched_code: Optional[str]
    # "class_match" | "fallback_unmatched" | "fallback_no_codes" | "unassigned"
    # "unassigned" is only produced by orders that declare a null fallback class: the
    # record matched no bucket and is to be EXCLUDED and counted, not swept into a class it
    # does not belong to. It carries label "" and label_index -1.
    matched_by: str
    all_codes: Tuple[str, ...]
    unknown_codes: Tuple[str, ...]

    @property
    def is_assigned(self) -> bool:
        return self.label_index >= 0


class ClassMap:
    """Loads and applies `class_map_7.json`."""

    def __init__(
        self,
        spec: dict,
        resolution_order: str = "clinical_specificity",
        vocabulary: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.spec = spec
        orders = spec["resolution_orders"]
        if resolution_order not in orders:
            raise KeyError(
                f"unknown resolution order {resolution_order!r}; "
                f"available: {sorted(orders)}"
            )
        self.resolution_order_name = resolution_order
        self.resolution_order: List[str] = list(orders[resolution_order])

        # Class names belong to the ordering, not to the corpus: canonical_v2 renames
        # OTHER -> VE because that bucket is the ventricular-ectopy family, never a
        # residue. Orders that declare no list keep the historical CLASS_NAMES, so every
        # previously published run reproduces unchanged.
        classes_key = f"classes_{resolution_order}"
        if classes_key in spec:
            self.classes = list(spec[classes_key])
            self.class_index = dict(
                spec.get(f"class_index_{resolution_order}")
                or {c: i for i, c in enumerate(self.classes)}
            )
        else:
            self.classes = list(spec["classes"])
            if tuple(self.classes) != tuple(CLASS_NAMES):
                raise ValueError(
                    f"class_map_7.json declares {self.classes} but config.CLASS_NAMES is "
                    f"{list(CLASS_NAMES)}; these must agree."
                )
            self.class_index = dict(spec["class_index"])

        unknown = [c for c in self.resolution_order if c not in self.class_index]
        if unknown:
            raise ValueError(
                f"resolution order {resolution_order!r} names {unknown}, which are not in "
                f"the class list {self.classes}."
            )
        if len(set(self.resolution_order)) != len(self.resolution_order):
            raise ValueError(
                f"resolution order {resolution_order!r} repeats a class: "
                f"{self.resolution_order}"
            )

        # A null fallback means "exclude and report" rather than "sweep into a class".
        fallback_key = f"fallback_class_{resolution_order}"
        self.fallback: Optional[str] = (
            spec[fallback_key] if fallback_key in spec else spec["fallback_class"]
        )
        if self.fallback is not None and self.fallback not in self.class_index:
            raise ValueError(
                f"fallback class {self.fallback!r} is not one of {self.classes}"
            )

        # A class may legitimately sit outside the precedence chain -- that is how the
        # fallback class is reached (rhythm_first omits OTHER, which then collects whatever
        # no rhythm claims). Any OTHER omission is unreachable: the class could never be
        # assigned to any record, which is a silent way to ship an empty class.
        orphans = [c for c in self.classes
                   if c not in self.resolution_order and c != self.fallback]
        if orphans:
            raise ValueError(
                f"resolution order {resolution_order!r} omits {orphans}, which are not the "
                f"fallback class ({self.fallback!r}); those classes could never be "
                "assigned to any record."
            )

        # An ordering may ship its own class definitions, so a published grouping can be
        # reproduced exactly without disturbing the default one. Convention:
        # "class_definitions_<order name>", a plain {class: [snomed, ...]} mapping.
        # Three sources, most authoritative first:
        #  1. acronym buckets resolved against the dataset's own
        #     ConditionNames_SNOMED-CT.csv -- the developers' mapping, no hand translation;
        #  2. a per-order SNOMED list committed here (best-effort translation);
        #  3. the default class_definitions.
        self.unmatched_acronyms: List[str] = []
        acronym_key = f"acronym_buckets_{resolution_order}"
        override_key = f"class_definitions_{resolution_order}"

        acronym_defs = None
        if acronym_key in spec and vocabulary:
            by_abbr: Dict[str, List[str]] = {}
            for code, meta in vocabulary.items():
                by_abbr.setdefault(str(meta.get("abbreviation", "")).upper(), []).append(code)
            acronym_defs = {}
            for cls in self.classes:
                codes: List[str] = []
                for acr in spec[acronym_key].get(cls, []):
                    hits = by_abbr.get(acr.upper(), [])
                    if not hits:
                        self.unmatched_acronyms.append(f"{cls}:{acr}")
                    codes.extend(hits)
                acronym_defs[cls] = {"snomed": codes}

        # A bucket often lists redundant alternates (["SR", "NSR"] where the vocabulary has
        # only "SR"), so an unmatched acronym is not by itself a problem. What matters is
        # whether any class ends up with nothing: that is when falling back protects a class
        # from being silently emptied.
        starved = (
            [c for c in self.resolution_order if not acronym_defs[c]["snomed"]]
            if acronym_defs is not None else []
        )
        if acronym_defs is not None and not starved:
            defs = acronym_defs
            self.definitions_source = f"{acronym_key} (resolved against the shipped vocabulary)"
            if self.unmatched_acronyms:
                self.definitions_source += (
                    f"; {len(self.unmatched_acronyms)} redundant alternate spelling(s) "
                    f"unmatched: {', '.join(self.unmatched_acronyms)}"
                )
        elif override_key in spec:
            # Acronyms did not fully resolve -- the vocabulary in use is not the one the
            # grouping was written against. Silently dropping the unmatched acronyms would
            # quietly shrink several classes, so fall back to the committed translation and
            # record why.
            defs = {c: {"snomed": list(spec[override_key].get(c, []))} for c in self.classes}
            self.definitions_source = (
                f"{override_key} (fallback: class(es) {starved} resolved to no codes from "
                f"{acronym_key} against this vocabulary; supply the dataset's own "
                "ConditionNames_SNOMED-CT.csv for the developers' mapping)"
                if acronym_defs is not None else override_key
            )
        else:
            defs = spec["class_definitions"]
            self.definitions_source = "class_definitions"

        self.code_to_class: Dict[str, str] = {}
        for cls in self.classes:
            for code in defs[cls]["snomed"]:
                code = str(code)
                # Several dataset acronyms share one code (LBBB / LBBBB / LFBBB are all
                # 164909002; IDC and IVB are both 698252002), so a bucket that lists the
                # alternate spellings resolves the same code more than once. That is a
                # duplicate, not a conflict; only a code claimed by two DIFFERENT classes
                # is an error.
                prior = self.code_to_class.get(code)
                if prior is not None and prior != cls:
                    raise ValueError(
                        f"SNOMED {code} assigned to both {prior} and {cls}; "
                        "a code may belong to at most one class."
                    )
                self.code_to_class[code] = cls
        # Codes in the vocabulary but in no bucket fall to `fallback`, or are excluded when
        # the ordering declares no fallback.
        self.class_to_codes: Dict[str, List[str]] = {
            cls: sorted({str(c) for c in defs[cls]["snomed"]}, key=int)
            for cls in self.classes
        }

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, data_dir: Path, resolution_order: str = "clinical_specificity") -> "ClassMap":
        path = Path(data_dir) / "class_map_7.json"
        if not path.is_file():
            raise FileNotFoundError(f"class map not found: {path}")
        # The vocabulary is loaded so acronym-bucket orderings can resolve against the
        # dataset developers' own ConditionNames_SNOMED-CT.csv when it is present. The map
        # must still load without one, so a missing vocabulary is not fatal.
        try:
            vocab = load_snomed_vocabulary(Path(data_dir))
        except Exception:  # noqa: BLE001
            vocab = None
        return cls(json.loads(path.read_text()), resolution_order=resolution_order,
                   vocabulary=vocab)

    # -- application -------------------------------------------------------
    def resolve(
        self,
        codes: Sequence[str],
        record_id: str = "",
        known_codes: Optional[set] = None,
    ) -> LabelResolution:
        codes = tuple(str(c) for c in codes)
        unknown = tuple(c for c in codes if known_codes is not None and c not in known_codes)

        def _unmatched(why: str) -> LabelResolution:
            if self.fallback is None:
                return LabelResolution(
                    record_id, "", -1, None, "unassigned", codes, unknown,
                )
            return LabelResolution(
                record_id, self.fallback, self.class_index[self.fallback],
                None, why, codes, unknown,
            )

        if not codes:
            return _unmatched("fallback_no_codes")

        code_set = set(codes)
        for cls in self.resolution_order:
            for code in self.class_to_codes[cls]:
                if code in code_set:
                    return LabelResolution(
                        record_id, cls, self.class_index[cls],
                        code, "class_match", codes, unknown,
                    )

        return _unmatched("fallback_unmatched")

    def fingerprint(self) -> Dict[str, object]:
        """Stable summary the tests assert against."""
        return {
            "classes": self.classes,
            # The NAME as well as the sequence: an artefact that records only the sequence
            # cannot be checked back against the shipped map, because nothing says which
            # entry of resolution_orders it came from.
            "resolution_order_name": self.resolution_order_name,
            "resolution_order": self.resolution_order,
            "definitions_source": self.definitions_source,
            "class_to_codes": {k: sorted(v, key=int) for k, v in self.class_to_codes.items()},
            "fallback_class": self.fallback,
        }


# ---------------------------------------------------------------------------
# SNOMED vocabulary
# ---------------------------------------------------------------------------
def load_snomed_vocabulary(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """Read `snomed_conditions.csv` -> {code: {abbreviation, full_name, ...}}.

    Also accepts the original Chapman `ConditionNames_SNOMED-CT.csv`
    (columns: Full Name, Acronym Name, Snomed_CT) if the user drops that in instead.
    """
    import csv

    data_dir = Path(data_dir)
    primary = data_dir / "snomed_conditions.csv"
    legacy = data_dir / "ConditionNames_SNOMED-CT.csv"

    # ConditionNames_SNOMED-CT.csv ships WITH Chapman-Shaoxing and is the data providers'
    # own acronym -> SNOMED mapping, so where both are present its acronyms win. It covers
    # 63 conditions; snomed_conditions.csv (PhysioNet/CinC 2021) covers 133, so the two are
    # merged rather than one replacing the other: any code the providers' file does not
    # mention still resolves, and every entry records where its abbreviation came from.
    if legacy.is_file() and primary.is_file():
        merged: Dict[str, Dict[str, str]] = {}
        with primary.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                merged[str(row["snomed_ct"]).strip()] = {
                    "abbreviation": row["abbreviation"].strip(),
                    "full_name": row["full_name"].strip(),
                    "chapman_count": row.get("chapman_count", "").strip(),
                    "source": "physionet_cinc_2021",
                }
        for code, meta in _read_legacy_conditions(legacy).items():
            existing = merged.get(code, {})
            merged[code] = {
                "abbreviation": meta["abbreviation"],          # providers' spelling wins
                "full_name": meta["full_name"] or existing.get("full_name", ""),
                "chapman_count": existing.get("chapman_count", ""),
                "source": "dataset_ConditionNames_SNOMED-CT",
            }
        return merged

    if primary.is_file():
        out: Dict[str, Dict[str, str]] = {}
        with primary.open(newline="") as fh:
            for row in csv.DictReader(fh):
                out[str(row["snomed_ct"]).strip()] = {
                    "abbreviation": row["abbreviation"].strip(),
                    "full_name": row["full_name"].strip(),
                    "chapman_count": row.get("chapman_count", "").strip(),
                }
        if not out:
            raise ValueError(f"{primary} contained no rows")
        return out

    if legacy.is_file():
        return _read_legacy_conditions(legacy)

    raise FileNotFoundError(
        f"no SNOMED vocabulary in {data_dir}: expected snomed_conditions.csv "
        f"or ConditionNames_SNOMED-CT.csv"
    )


def _read_legacy_conditions(legacy: Path) -> Dict[str, Dict[str, str]]:
    """Read the dataset's own ConditionNames_SNOMED-CT.csv.

    Columns are 'Acronym Name', 'Full Name', 'Snomed_CT'. The file is UTF-8 with a BOM, so
    it is opened utf-8-sig -- otherwise the first header becomes '\ufeffAcronym Name' and
    the column lookup silently misses.
    """
    import csv

    out: Dict[str, Dict[str, str]] = {}
    if True:
        with legacy.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields = {c.strip().lower().replace(" ", "_"): c for c in (reader.fieldnames or [])}
            col_code = next((fields[k] for k in fields if "snomed" in k), None)
            col_acr = next((fields[k] for k in fields if "acronym" in k), None)
            col_name = next((fields[k] for k in fields if "full" in k or "name" == k), None)
            if col_code is None or col_acr is None:
                raise ValueError(
                    f"{legacy} is missing a SNOMED or acronym column; "
                    f"found {reader.fieldnames}"
                )
            for row in reader:
                out[str(row[col_code]).strip()] = {
                    "abbreviation": row[col_acr].strip(),
                    "full_name": (row[col_name].strip() if col_name else ""),
                    "chapman_count": "",
                    "source": "dataset_ConditionNames_SNOMED-CT",
                }
    return out


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------
def parse_dx_codes(header_path: Path) -> List[str]:
    """Extract SNOMED CT codes from a WFDB header's `#Dx:` line.

    Deliberately strict: only the `#Dx:` line is read. The notebook this task derives from
    scanned every comment line containing "SNOMED" or "Dx" and kept any >=5-digit number,
    which also swallows `#Age: 85` style fields on some corpora and silently mislabels.
    """
    text = Path(header_path).read_text(errors="replace")
    for line in text.splitlines():
        m = _DX_RE.match(line.strip())
        if m:
            codes = [c for c in _CODE_RE.findall(m.group(1))]
            # Preserve order, drop duplicates.
            seen, out = set(), []
            for c in codes:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
    return []


@lru_cache(maxsize=1)
def _cached_map(data_dir_str: str, order: str) -> ClassMap:
    return ClassMap.load(Path(data_dir_str), resolution_order=order)


def get_class_map(data_dir: Path, resolution_order: str = "clinical_specificity") -> ClassMap:
    """Load the class map AND publish its class names as the process-wide active list.

    Every downstream artefact (confusion matrix axes, per-class metric keys, XAI CSV rows)
    is keyed by class name, so the name list has to follow the resolution order rather than
    a module constant. This is the single point where it is set.
    """
    cmap = _cached_map(str(Path(data_dir).resolve()), resolution_order)
    set_active_class_names(cmap.classes)
    return cmap


# ---------------------------------------------------------------------------
# PTB-XL (optional, cross-dataset generalisation only)
# ---------------------------------------------------------------------------
# PTB-XL ships SCP-ECG statements, not SNOMED. Used only when --eval-ptbxl is passed.
SCP_TO_CLASS: Dict[str, str] = {
    "SR": "NSR", "SARRH": "NSR",
    "AFIB": "AFIB", "AFLT": "AFIB",
    "SBRAD": "SB",
    "STACH": "ST",
    "SVTAC": "SVT", "PSVT": "SVT",
    "CRBBB": "CD", "IRBBB": "CD", "CLBBB": "CD", "ILBBB": "CD",
    "LAFB": "CD", "LPFB": "CD", "IVCD": "CD", "WPW": "CD",
    "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
    "PACE": "CD",
    "BIGU": "OTHER", "TRIGU": "OTHER", "PVC": "OTHER", "PAC": "OTHER",
}

#: canonical_v2 renames OTHER -> VE and moves pre-excitation (WPW) into CD, so the SCP
#: mapping has to follow. PAC is an atrial premature beat, not ventricular ectopy, so under
#: v2 it maps to nothing and the record is excluded rather than mislabelled.
SCP_TO_CLASS_V2: Dict[str, str] = {
    "SR": "NSR", "SARRH": "NSR",
    "AFIB": "AFIB", "AFLT": "AFIB",
    "SBRAD": "SB",
    "STACH": "ST",
    "SVTAC": "SVT", "PSVT": "SVT",
    "CRBBB": "CD", "IRBBB": "CD", "CLBBB": "CD", "ILBBB": "CD",
    "LAFB": "CD", "LPFB": "CD", "IVCD": "CD", "WPW": "CD",
    "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
    "PACE": "CD",
    "BIGU": "VE", "TRIGU": "VE", "PVC": "VE",
}


def scp_mapping(resolution_order: str) -> Dict[str, str]:
    """The SCP-ECG -> class mapping matching a resolution order (PTB-XL only)."""
    return SCP_TO_CLASS_V2 if resolution_order == "canonical_v2" else SCP_TO_CLASS


__all__ = [
    "ClassMap", "LabelResolution", "load_snomed_vocabulary", "parse_dx_codes",
    "get_class_map", "SCP_TO_CLASS", "SCP_TO_CLASS_V2", "scp_mapping",
]
