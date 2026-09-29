"""SNOMED CT vocabulary, class map, and multi-label -> single-label resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecgvit.config import CLASS_NAMES
from ecgvit.labels import ClassMap, get_class_map, load_snomed_vocabulary, parse_dx_codes


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def test_vocabulary_loads_and_codes_are_numeric(data_dir):
    vocab = load_snomed_vocabulary(data_dir)
    assert len(vocab) >= 100, "SNOMED vocabulary looks truncated"
    for code, meta in vocab.items():
        assert code.isdigit(), f"non-numeric SNOMED id: {code!r}"
        assert meta["abbreviation"], f"{code} has no abbreviation"


@pytest.mark.parametrize(
    "code,concept_words",
    [
        ("164889003", ["atrial", "fibrillation"]),
        ("164890007", ["atrial", "flutter"]),
        ("426177001", ["sinus", "bradycardia"]),
        ("426783006", ["sinus", "rhythm"]),
        ("427084000", ["sinus", "tachycardia"]),
        ("426761007", ["supraventricular", "tachycardia"]),
        ("59118001",  ["right", "bundle", "branch"]),
    ],
)
def test_known_snomed_codes_denote_the_expected_concept(data_dir, code, concept_words):
    """Assert the CONCEPT, not the abbreviation string.

    Abbreviations are vocabulary-dependent and the two vocabularies in play disagree, so
    pinning a spelling here would just encode whichever file happened to load. The SNOMED
    code and the concept it denotes are stable; the abbreviation is not.
    """
    vocab = load_snomed_vocabulary(data_dir)
    assert code in vocab, f"{code} missing from the vocabulary"
    name = vocab[code]["full_name"].lower()
    missing = [w for w in concept_words if w not in name]
    assert not missing, f"{code} is '{name}', expected a concept containing {concept_words}"


def test_the_af_abbreviation_collision_is_handled(data_dir):
    """A genuine hazard, asserted so it cannot be forgotten.

    In the dataset developers' ConditionNames_SNOMED-CT.csv, 'AF' is ATRIAL FLUTTER and
    'AFIB' is atrial fibrillation. In the PhysioNet/CinC 2021 tables, 'AF' is atrial
    FIBRILLATION and 'AFL' is flutter. Code that mixes the two vocabularies silently swaps
    the two conditions. Both resolve to the same AFIB class here, so nothing downstream is
    affected -- but any analysis that separates fibrillation from flutter must pin the
    vocabulary first.
    """
    vocab = load_snomed_vocabulary(data_dir)
    fib, flut = vocab["164889003"], vocab["164890007"]
    assert "fibrillation" in fib["full_name"].lower()
    assert "flutter" in flut["full_name"].lower()
    if flut["abbreviation"].upper() == "AF":
        assert flut["source"] == "dataset_ConditionNames_SNOMED-CT", (
            "'AF' denotes flutter only in the dataset's own vocabulary"
        )
    cmap = get_class_map(data_dir, "canonical_hmgmedformer")
    assert cmap.resolve(["164889003"]).label == "AFIB"
    assert cmap.resolve(["164890007"]).label == "AFIB"


def test_vocabulary_records_the_provenance_of_every_entry(data_dir):
    vocab = load_snomed_vocabulary(data_dir)
    sources = {m.get("source") for m in vocab.values()}
    assert sources <= {"dataset_ConditionNames_SNOMED-CT", "physionet_cinc_2021"}
    assert None not in sources, "an entry has no recorded source"


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "order,snapshot",
    [
        ("rhythm_first", "snomed_class_map.expected.json"),
        ("canonical_v2", "snomed_class_map_canonical_v2.expected.json"),
    ],
)
def test_class_map_matches_expected_snapshot(data_dir, expected_dir, order, snapshot):
    """Exact-match guard. The class map is the single most consequential preprocessing
    decision in the pipeline; it must not drift without a deliberate snapshot update.

    canonical_v2 is snapshotted alongside the historical default because its buckets are
    the correction that the results review turned on -- an accidental edit to the ST bucket
    would silently reintroduce ~1,100 mislabelled bradycardic records."""
    cmap = get_class_map(data_dir, order)
    expected = json.loads((expected_dir / snapshot).read_text())
    assert cmap.fingerprint() == expected


def test_class_names_agree_across_config_and_data(data_dir):
    cmap = get_class_map(data_dir)
    assert tuple(cmap.classes) == tuple(CLASS_NAMES)
    assert set(cmap.class_index) == set(CLASS_NAMES)
    assert sorted(cmap.class_index.values()) == list(range(len(CLASS_NAMES)))


def test_no_snomed_code_belongs_to_two_classes(data_dir):
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    seen = {}
    for cls, d in spec["class_definitions"].items():
        for code in d["snomed"]:
            assert code not in seen, f"{code} in both {seen[code]} and {cls}"
            seen[code] = cls


def test_unmatched_records_go_to_the_fallback_or_are_excluded(data_dir):
    """Every ordering must do something DEFINED with a record it cannot classify.

    Two defined behaviours, and no third:

    * a fallback class is declared -- the record lands there and says so
      (`fallback_unmatched` / `fallback_no_codes`);
    * the fallback is null (canonical_v2) -- the record is marked `unassigned` with
      label_index -1, and `build_index` drops and counts it.

    What must never happen is a record silently acquiring a class whose definition it does
    not meet. canonical_v2 declares no fallback precisely because its buckets are positive
    clinical definitions: a record carrying only, say, a wandering atrial pacemaker code is
    not ventricular ectopy, and calling it VE to avoid an empty cell would be a fabrication.
    """
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    assert spec["fallback_class"] == "OTHER"
    for name in spec["resolution_orders"]:
        cmap = get_class_map(data_dir, name)
        unmatched = cmap.resolve(["999999999"])
        empty = cmap.resolve([])
        if cmap.fallback is None:
            for res in (unmatched, empty):
                assert res.matched_by == "unassigned", name
                assert res.label == "" and res.label_index == -1, name
                assert res.is_assigned is False, name
        else:
            assert unmatched.label == cmap.fallback, name
            assert unmatched.matched_by == "fallback_unmatched", name
            assert empty.matched_by == "fallback_no_codes", name
            assert unmatched.is_assigned is True, name


def test_canonical_v2_corrects_the_st_bucket(data_dir):
    """The ST class must be sinus tachycardia and nothing else.

    canonical_hmgmedformer merged five repolarisation MORPHOLOGY findings (TWC, STTC, STDD,
    STTU, STE) into the sinus-tachycardia RHYTHM class on the shared 'ST' acronym prefix.
    Because ST outranks SB and NSR in the precedence chain, that merge relabelled roughly
    1,100 bradycardic records as tachycardia. This test pins the fix.
    """
    v1 = get_class_map(data_dir, "canonical_hmgmedformer")
    v2 = get_class_map(data_dir, "canonical_v2")

    SINUS_TACHYCARDIA = "427084000"
    REPOLARISATION = {
        "164934002",  # TWC   T wave Change
        "428750005",  # STTC  ST-T Change
        "429622005",  # STDD  ST drop down
        "164931005",  # STTU  ST tilt up
        "164930006",  # STE   ST extension
    }
    # The defect, asserted so the regression is visible if anyone "fixes" v1.
    assert REPOLARISATION <= set(v1.class_to_codes["ST"])
    # The correction.
    assert v2.class_to_codes["ST"] == [SINUS_TACHYCARDIA]
    assert not REPOLARISATION & set(c for cs in v2.class_to_codes.values() for c in cs)

    # A bradycardic record that also carries a T-wave change is SB under v2, ST under v1.
    codes = ["426177001", "164934002"]
    assert v1.resolve(codes).label == "ST"
    assert v2.resolve(codes).label == "SB"


def test_canonical_v2_renames_other_to_ve_and_keeps_the_index(data_dir):
    """OTHER becomes VE, in the same index position, with the ectopy codes intact."""
    v1 = get_class_map(data_dir, "canonical_hmgmedformer")
    v2 = get_class_map(data_dir, "canonical_v2")
    assert v2.classes == ["NSR", "AFIB", "SB", "ST", "SVT", "CD", "VE"]
    assert "OTHER" not in v2.classes
    assert v2.class_index["VE"] == v1.class_index["OTHER"] == 6
    # Ventricular ectopy: VPB, VEB (two codes), VB, VFW, VET.
    assert {"17338001", "75532003", "11157007", "13640000", "251180001"} <= set(
        v2.class_to_codes["VE"]
    )
    # Pre-excitation is an accessory-pathway conduction abnormality, not an ectopic beat.
    assert "195060002" in v2.class_to_codes["CD"]
    assert "195060002" not in v2.class_to_codes["VE"]


def test_canonical_v2_cleans_the_cd_afib_and_svt_buckets(data_dir):
    v2 = get_class_map(data_dir, "canonical_v2")
    # Precordial rotation is an axis descriptor, not a conduction disturbance.
    for rotation in ("251198002", "251199005"):     # CR, CCR
        assert rotation not in v2.class_to_codes["CD"]
    # A wandering atrial pacemaker is neither fibrillation nor flutter.
    for wap in ("17366009", "195101003"):           # SAAWR / WAVN
        assert wap not in v2.class_to_codes["AFIB"]
    assert v2.class_to_codes["AFIB"] == ["164889003", "164890007"]
    # A junctional premature beat is not a tachycardia; AVNRT is.
    assert "251164006" not in v2.class_to_codes["SVT"]          # JPT
    assert {"233896004", "251166008"} & set(v2.class_to_codes["SVT"])  # AVNRT
    # Conduction codes the earlier bucket missed.
    assert {"28189009", "74390002"} <= set(v2.class_to_codes["CD"])    # 2AVB2, WPW


def test_canonical_v2_findings_must_outrank_rhythms(data_dir):
    """CD and VE come first, and that is forced, not chosen.

    Every Chapman record carries exactly one rhythm code, so an order that claims a rhythm
    before a finding leaves CD and VE with zero records -- the finding is always absorbed.
    That is the argument for the multi-label formulation, and it is checked here rather
    than asserted in prose.
    """
    v2 = get_class_map(data_dir, "canonical_v2")
    assert v2.resolution_order[:2] == ["CD", "VE"]
    # Atrial fibrillation with right bundle branch block: single-label precedence must pick
    # one, and picking CD is what produces the CD<->AFIB confusion in the results.
    assert v2.resolve(["164889003", "59118001"]).label == "CD"


def test_legacy_orders_leave_other_as_a_pure_residue(data_dir):
    """The three orderings that predate the ectopy definition must not match OTHER
    positively, so their behaviour is unchanged."""
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    for name in ("conduction_first", "rhythm_first", "paper_reported"):
        assert "OTHER" not in spec["resolution_orders"][name]


def test_all_class_codes_exist_in_vocabulary(data_dir):
    vocab = set(load_snomed_vocabulary(data_dir))
    cmap = get_class_map(data_dir)
    for cls, codes in cmap.class_to_codes.items():
        unknown = [c for c in codes if c not in vocab]
        assert not unknown, f"{cls} references codes absent from the vocabulary: {unknown}"


def test_unknown_resolution_order_is_rejected(data_dir):
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    with pytest.raises(KeyError):
        ClassMap(spec, resolution_order="does_not_exist")


# ---------------------------------------------------------------------------
# Resolution semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "codes,expected,why",
    [
        (["426783006"], "NSR", "plain sinus rhythm"),
        (["164889003"], "AFIB", "atrial fibrillation"),
        (["164890007"], "AFIB", "flutter groups with AF, per Zheng et al."),
        (["426177001"], "SB", "sinus bradycardia"),
        (["427084000"], "ST", "sinus tachycardia"),
        (["426761007"], "SVT", "supraventricular tachycardia"),
        (["59118001"], "CD", "RBBB is a conduction disturbance"),
        (["164934002"], "OTHER", "T-wave abnormality alone belongs to no rhythm class"),
        ([], "OTHER", "no codes at all"),
        (["999999999"], "OTHER", "code outside the vocabulary"),
        (["426783006", "164934002"], "NSR", "sinus rhythm wins over a T-wave finding"),
        (["164889003", "59118001"], "AFIB", "AFIB precedes CD in rhythm_first"),
        (["426177001", "59118001"], "SB", "a rhythm class precedes CD"),
    ],
)
def test_resolution_order_semantics(data_dir, codes, expected, why):
    cmap = get_class_map(data_dir, "rhythm_first")
    assert cmap.resolve(codes).label == expected, why


@pytest.mark.parametrize(
    "codes,expected,why",
    [
        (["426177001", "59118001"], "CD", "SB + RBBB -> CD; conduction outranks rhythm"),
        (["164889003", "59118001"], "CD", "AFIB + RBBB -> CD"),
        (["426783006", "270492004"], "CD", "sinus rhythm + 1st degree AV block -> CD"),
        (["426177001"], "SB", "no conduction code, so the rhythm still wins"),
        (["164889003"], "AFIB", "no conduction code"),
        (["164934002"], "OTHER", "neither rhythm nor conduction"),
    ],
)
def test_conduction_first_claims_every_conduction_record(data_dir, codes, expected, why):
    """The default ordering. Measured on the real corpus: any ordering that lets a rhythm
    class win first leaves CD with 59 records total, because conduction findings in
    Chapman-Shaoxing nearly always co-occur with a rhythm code. This ordering gives CD the
    records it is named for, at the documented cost that SB/AFIB lose their
    conduction-affected members."""
    assert get_class_map(data_dir, "conduction_first").resolve(codes).label == expected, why


def test_the_three_orders_disagree_on_a_rhythm_plus_conduction_record(data_dir):
    """SB + RBBB is the record that separates all three orderings. If this ever collapses,
    docs/CLASS_MAPPING.md is describing a distinction that no longer exists."""
    codes = ["426177001", "59118001"]
    assert get_class_map(data_dir, "conduction_first").resolve(codes).label == "CD"
    assert get_class_map(data_dir, "rhythm_first").resolve(codes).label == "SB"
    assert get_class_map(data_dir, "paper_reported").resolve(codes).label == "SB"


def test_every_shipped_order_covers_the_six_rhythm_and_conduction_classes(data_dir):
    """An order that omits a class silently makes it unreachable. OTHER is optional --
    it is the fallback regardless -- but the other six must always be listed."""
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    required = set(CLASS_NAMES) - {"OTHER"}
    for name, order in spec["resolution_orders"].items():
        assert required <= set(order), f"{name} cannot reach {required - set(order)}"
        assert len(order) == len(set(order)), f"{name} lists a class twice"


@pytest.mark.parametrize(
    "codes,expected,why",
    [
        (["426177001", "17338001"], "OTHER", "SB + ventricular premature beats -> OTHER"),
        (["426783006", "284470004"], "OTHER", "sinus rhythm + PAC -> OTHER"),
        (["426177001", "59118001", "17338001"], "CD", "conduction outranks ectopy"),
        (["426177001"], "SB", "no ectopy and no conduction, so the rhythm wins"),
        (["164889003"], "AFIB", "plain AF"),
        (["164934002"], "OTHER", "unmatched code still falls back to OTHER"),
    ],
)
def test_clinical_specificity_populates_other(data_dir, codes, expected, why):
    """The default ordering, and the only one under which all seven classes are non-empty
    on Chapman-Shaoxing. Ectopy and premature beats are beat-level findings that always
    coexist with a rhythm label, so OTHER only fills when given precedence over rhythm."""
    assert get_class_map(data_dir, "clinical_specificity").resolve(codes).label == expected, why


def test_other_is_unreachable_by_matching_in_the_legacy_orders(data_dir):
    """The same record that is OTHER under the default is a rhythm class under the others.
    This is the measured reason OTHER came out empty on the real corpus."""
    codes = ["426177001", "17338001"]   # sinus bradycardia + ventricular premature beats
    assert get_class_map(data_dir, "clinical_specificity").resolve(codes).label == "OTHER"
    for name in ("conduction_first", "rhythm_first", "paper_reported"):
        assert get_class_map(data_dir, name).resolve(codes).label == "SB"


def test_default_order_is_the_one_documented_in_the_class_map(data_dir):
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    assert spec["default_resolution_order"] in spec["resolution_orders"]
    from ecgvit.config import PipelineConfig

    assert PipelineConfig().resolution_order == spec["default_resolution_order"], (
        "the pipeline default and the class map's declared default disagree"
    )


def test_rhythm_first_and_paper_reported_differ_where_documented(data_dir):
    """A sinus-rhythm record that also carries RBBB is the case the two orders disagree on.
    That disagreement is the whole reason both orders exist -- if it ever vanishes, the
    documentation in docs/CLASS_MAPPING.md is wrong."""
    codes = ["426783006", "59118001"]
    assert get_class_map(data_dir, "rhythm_first").resolve(codes).label == "CD"
    assert get_class_map(data_dir, "paper_reported").resolve(codes).label == "NSR"


@pytest.mark.parametrize(
    "order,expected",
    [("conduction_first", "CD"), ("rhythm_first", "SB"), ("paper_reported", "SB")],
)
def test_resolution_is_order_independent_within_a_record(data_dir, order, expected):
    """The order codes appear in the HEADER must not change the label -- only the
    resolution order may. Checked for every shipped ordering, since a lookup that
    accidentally depended on header order would be invisible with just one."""
    cmap = get_class_map(data_dir, order)
    a = cmap.resolve(["164934002", "426177001", "59118001"]).label
    b = cmap.resolve(["59118001", "164934002", "426177001"]).label
    c = cmap.resolve(["426177001", "59118001", "164934002"]).label
    assert a == b == c == expected


def test_resolution_reports_provenance(data_dir):
    cmap = get_class_map(data_dir)
    r = cmap.resolve(["164889003", "164934002"], "JS00001", known_codes={"164889003"})
    assert r.label == "AFIB"
    assert r.matched_code == "164889003"
    assert r.matched_by == "class_match"
    assert r.unknown_codes == ("164934002",)
    assert cmap.resolve([]).matched_by == "fallback_no_codes"
    assert cmap.resolve(["164934002"]).matched_by == "fallback_unmatched"


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------
def test_parse_dx_from_real_header_shape(tmp_path):
    hea = tmp_path / "JS00001.hea"
    hea.write_text(
        "JS00001 12 500 5000 23-Mar-2021 20:20:47\n"
        "JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I\n"
        "#Age: 85\n#Sex: Male\n#Dx: 164889003,59118001,164934002\n#Rx: Unknown\n"
    )
    assert parse_dx_codes(hea) == ["164889003", "59118001", "164934002"]


def test_parse_dx_ignores_age_and_other_comment_fields(tmp_path):
    """Regression guard. A looser parser that scans every comment line for long numbers
    picks up dates and ids and mislabels records."""
    hea = tmp_path / "x.hea"
    hea.write_text(
        "x 12 500 5000 23-Mar-2021 20:20:47\n"
        "#Age: 85\n"
        "#Sex: Male\n"
        "#Hx: prior admission 4268310000 not a diagnosis\n"
        "#Dx: 426177001\n"
    )
    assert parse_dx_codes(hea) == ["426177001"]


def test_parse_dx_deduplicates_preserving_order(tmp_path):
    hea = tmp_path / "x.hea"
    hea.write_text("x 12 500 5000\n#Dx: 426177001,59118001,426177001\n")
    assert parse_dx_codes(hea) == ["426177001", "59118001"]


def test_parse_dx_absent_returns_empty(tmp_path):
    hea = tmp_path / "x.hea"
    hea.write_text("x 12 500 5000\n#Age: 55\n#Sex: Male\n")
    assert parse_dx_codes(hea) == []


def test_fixture_corpus_labels_resolve_as_designed(fixture_corpus, fixture_labels, data_dir):
    cmap = get_class_map(data_dir, "rhythm_first")
    for rid, expected in fixture_labels.items():
        codes = parse_dx_codes(fixture_corpus / f"{rid}.hea")
        assert codes, f"{rid} has no #Dx codes"
        assert cmap.resolve(codes, rid).label == expected, (
            f"{rid} carries {codes} and should resolve to {expected}"
        )


def test_every_class_is_represented_in_the_fixtures(fixture_labels):
    assert set(fixture_labels.values()) == set(CLASS_NAMES)


# ---------------------------------------------------------------------------
# The published HMGMedFormer grouping
# ---------------------------------------------------------------------------
def test_hmgmedformer_order_uses_its_own_definitions(data_dir):
    """An ordering may ship its own class definitions so a published grouping is
    reproduced exactly without perturbing the default one."""
    cm = get_class_map(data_dir, "canonical_hmgmedformer")
    assert cm.definitions_source.startswith("class_definitions_canonical_hmgmedformer") or \
        cm.definitions_source.startswith("acronym_buckets_canonical_hmgmedformer")
    assert get_class_map(data_dir, "clinical_specificity").definitions_source == \
        "class_definitions"


def test_no_class_is_left_empty_by_acronym_resolution(data_dir):
    """A bucket may list redundant alternate spellings, so an unmatched acronym is fine.
    A class resolving to NO codes is not: it would be silently unreachable. The loader
    falls back to the committed translation in that case, and either way every class in
    the ordering must end up with at least one code."""
    cm = get_class_map(data_dir, "canonical_hmgmedformer")
    empty = [c for c in cm.resolution_order if not cm.class_to_codes[c]]
    assert not empty, f"{empty} resolved to no codes under {cm.definitions_source}"
    assert "fallback" not in cm.definitions_source or "ConditionNames" in cm.definitions_source


@pytest.mark.parametrize(
    "codes,default_label,hmg_label,why",
    [
        (["17366009"],  "SVT",   "AFIB",  "to_canonical puts SAAWR with AFIB, not SVT"),
        (["164931005"], "OTHER", "ST",    "to_canonical's ST absorbs ST elevation"),
        (["429622005"], "OTHER", "ST",    "and ST depression"),
        (["251199005"], "OTHER", "CD",    "to_canonical puts CCR rotation in CD"),
        (["427393009"], "NSR",   "OTHER", "to_canonical's NSR is SR only, so SA falls out"),
        (["426648003"], "SVT",   "OTHER", "to_canonical's SVT omits junctional tachycardia"),
    ],
)
def test_hmgmedformer_grouping_differs_where_documented(
    data_dir, codes, default_label, hmg_label, why
):
    """Every documented divergence between the published grouping and this repository's
    default. If one of these collapses, the note in class_map_7.json is describing a
    difference that no longer exists."""
    assert get_class_map(data_dir, "clinical_specificity").resolve(codes).label == default_label
    assert get_class_map(data_dir, "canonical_hmgmedformer").resolve(codes).label == hmg_label, why


def test_hmgmedformer_definitions_have_no_overlapping_codes(data_dir):
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    seen = {}
    for cls, codes in spec["class_definitions_canonical_hmgmedformer"].items():
        for c in codes:
            assert c not in seen, f"{c} in both {seen[c]} and {cls}"
            seen[c] = cls


def test_ambiguous_acronym_aliases_are_flagged_not_guessed(data_dir):
    """Six of the published acronyms map to more than one SNOMED concept in this
    vocabulary. They must be recorded as ambiguous rather than silently assigned."""
    spec = json.loads((data_dir / "class_map_7.json").read_text())
    amb = spec["chapman_acronym_aliases"]["ambiguous_not_assigned"]
    assert set(amb) == {"STTC", "STTU", "TWC", "JPT", "VET", "LFBBB"}
    assigned = set(spec["chapman_acronym_aliases"]["confident"])
    assert not (assigned & set(amb)), "an acronym is both confident and ambiguous"
    for k, v in amb.items():
        assert len(v["candidates"]) >= 1 and v["reason"]
