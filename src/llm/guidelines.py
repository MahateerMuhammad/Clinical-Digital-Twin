"""
src/llm/guidelines.py
─────────────────────
Level 1 evidence tier: a curated, offline, versioned guideline corpus.

The RAG previously emitted the label "Level 1: Clinical Practice Guidelines" and a
"No Level 1 Guideline Retrieved" refusal, but had **no guideline source of any
kind**. This module supplies one.

Provenance policy
-----------------
Every record is a *paraphrased summary* of a published recommendation, not
verbatim guideline text, and is marked ``verbatim=False``. Each carries the
source society, document, year and a URL so a clinician can verify the wording
against the primary document. Records are keyed to canonical concepts from
:mod:`src.llm.terminology`, so retrieval works on normalised concepts rather
than free-text substrings.

This corpus is deliberately small and conservative. It is a scaffold to be
extended and clinically reviewed — ``review_status`` records that state per
record, and unreviewed records can be filtered out by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["GuidelineRecord", "GUIDELINE_CORPUS", "retrieve_guidelines", "corpus_stats"]

CORPUS_VERSION = "2026.07.1"


@dataclass(frozen=True)
class GuidelineRecord:
    doc_id: str
    concepts: Tuple[str, ...]
    society: str
    document: str
    year: int
    section: str
    recommendation: str
    strength: str                  # e.g. "strong recommendation, moderate quality"
    url: str
    verbatim: bool = False
    review_status: str = "unreviewed"   # unreviewed | clinician_reviewed
    keywords: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def citation(self) -> str:
        return f"[{self.society} {self.document} {self.year}, {self.section}]"

    def to_doc(self) -> dict:
        """Render in the same shape the RAG uses for other evidence documents."""
        return {
            "doc_id": self.doc_id,
            "citation": self.citation,
            "title": f"{self.society} {self.document} ({self.year}): {self.section}",
            "category": "Clinical Practice Guideline (curated offline corpus)",
            "evidence_level": "Level 1: Clinical Practice Guidelines",
            "text": self.recommendation,
            "strength": self.strength,
            "url": self.url,
            "verbatim": self.verbatim,
            "review_status": self.review_status,
            "corpus_version": CORPUS_VERSION,
            "provenance": "paraphrased summary; verify against source document",
        }


def _g(doc_id, concepts, society, document, year, section, rec, strength, url, keywords=()):
    return GuidelineRecord(doc_id, tuple(concepts), society, document, year, section,
                           rec, strength, url, keywords=tuple(keywords))


GUIDELINE_CORPUS: Tuple[GuidelineRecord, ...] = (
    # ── sepsis ───────────────────────────────────────────────────────────
    _g("SSC2021-ABX", ["sepsis"], "SCCM/ESICM", "Surviving Sepsis Campaign", 2021,
       "Initial Resuscitation / Antimicrobials",
       "For adults with possible septic shock, administer antimicrobials immediately, "
       "ideally within 1 hour of recognition. Obtain blood cultures before antimicrobials "
       "where this does not materially delay administration.",
       "strong recommendation, moderate quality of evidence",
       "https://www.sccm.org/survivingsepsiscampaign",
       ["antibiotic", "antimicrobial", "blood culture", "timing"]),
    _g("SSC2021-FLUID", ["sepsis"], "SCCM/ESICM", "Surviving Sepsis Campaign", 2021,
       "Initial Resuscitation",
       "For sepsis-induced hypoperfusion or septic shock, at least 30 mL/kg of intravenous "
       "crystalloid should be given within the first 3 hours, with subsequent fluid guided by "
       "dynamic measures of fluid responsiveness rather than fixed volumes.",
       "weak recommendation, low quality of evidence",
       "https://www.sccm.org/survivingsepsiscampaign",
       ["crystalloid", "fluid", "resuscitation", "responsiveness"]),
    _g("SSC2021-MAP", ["sepsis"], "SCCM/ESICM", "Surviving Sepsis Campaign", 2021,
       "Haemodynamic Management",
       "Norepinephrine is the first-line vasopressor for septic shock, targeting an initial "
       "mean arterial pressure of 65 mmHg. Vasopressin may be added to reduce norepinephrine "
       "requirements rather than escalating norepinephrine indefinitely.",
       "strong recommendation, moderate quality of evidence",
       "https://www.sccm.org/survivingsepsiscampaign",
       ["norepinephrine", "vasopressor", "map", "vasopressin"]),

    # ── AKI ──────────────────────────────────────────────────────────────
    _g("KDIGO2012-AKI-NEPHROTOX", ["aki"], "KDIGO", "Clinical Practice Guideline for AKI", 2012,
       "Section 3: Prevention and Treatment",
       "In patients with or at risk of AKI, discontinue nephrotoxic agents where possible, "
       "ensure volume status and perfusion pressure are optimised, and monitor serum "
       "creatinine and urine output closely. Avoid hyperglycaemia.",
       "graded recommendation (1B/2C by sub-item)",
       "https://kdigo.org/guidelines/acute-kidney-injury/",
       ["nephrotoxic", "creatinine", "urine output", "volume"]),
    _g("KDIGO2012-AKI-RRT", ["aki"], "KDIGO", "Clinical Practice Guideline for AKI", 2012,
       "Section 5: Renal Replacement Therapy",
       "Initiate renal replacement therapy emergently for life-threatening changes in fluid, "
       "electrolyte and acid-base balance. Base the decision on the broader clinical context "
       "rather than the serum creatinine or urea value alone.",
       "not graded / expert consensus",
       "https://kdigo.org/guidelines/acute-kidney-injury/",
       ["dialysis", "rrt", "crrt", "electrolyte", "acid base"]),
    _g("KDIGO2012-AKI-DOSING", ["aki"], "KDIGO", "Clinical Practice Guideline for AKI", 2012,
       "Section 3.9: Drug Dosing",
       "Adjust renally-cleared drug doses for the current level of kidney function. Serum "
       "creatinine-based estimates are unreliable in non-steady-state AKI, so dose "
       "adjustment should account for changing function rather than a single eGFR value.",
       "not graded / expert consensus",
       "https://kdigo.org/guidelines/acute-kidney-injury/",
       ["dose", "dosing", "renal clearance", "egfr", "adjustment"]),

    # ── heart failure ────────────────────────────────────────────────────
    _g("AHA2022-HF-DIURESIS", ["heart_failure"], "ACC/AHA/HFSA",
       "Guideline for the Management of Heart Failure", 2022,
       "Acute Decompensated HF",
       "Patients admitted with acute decompensated heart failure and evidence of volume "
       "overload should receive intravenous loop diuretics, with dose titrated to achieve "
       "decongestion and urine output and renal function monitored during therapy.",
       "Class 1, Level of Evidence B-NR",
       "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
       ["furosemide", "loop diuretic", "decongestion", "volume overload"]),
    _g("AHA2022-HF-CARDIORENAL", ["heart_failure", "aki"], "ACC/AHA/HFSA",
       "Guideline for the Management of Heart Failure", 2022,
       "Cardiorenal Considerations",
       "A modest rise in serum creatinine during effective decongestion does not by itself "
       "mandate stopping diuresis. Assess volume status directly rather than treating the "
       "creatinine change in isolation.",
       "expert consensus statement",
       "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063",
       ["cardiorenal", "creatinine", "diuresis", "worsening renal function"]),

    # ── DKA / hyperglycaemic crisis ──────────────────────────────────────
    _g("ADA-DKA-POTASSIUM", ["dka", "hyperkalemia"], "ADA",
       "Hyperglycaemic Crises in Adult Patients with Diabetes", 2009,
       "Potassium Replacement",
       "Withhold insulin if serum potassium is below 3.3 mEq/L until potassium is repleted, "
       "because insulin drives potassium intracellularly and can precipitate life-threatening "
       "hypokalaemia and arrhythmia.",
       "consensus statement",
       "https://diabetesjournals.org/care/article/32/7/1335/28675",
       ["potassium", "insulin", "hypokalemia", "replacement"]),
    _g("ADA-DKA-FLUID", ["dka"], "ADA",
       "Hyperglycaemic Crises in Adult Patients with Diabetes", 2009,
       "Fluid Therapy",
       "Begin isotonic crystalloid resuscitation to restore intravascular volume, then adjust "
       "tonicity according to corrected serum sodium. Add dextrose to the infusion once "
       "glucose falls to approximately 200 mg/dL while ketoacidosis is still resolving.",
       "consensus statement",
       "https://diabetesjournals.org/care/article/32/7/1335/28675",
       ["crystalloid", "saline", "dextrose", "sodium", "anion gap"]),

    # ── hyperkalaemia ────────────────────────────────────────────────────
    _g("HYPERK-STABILISE", ["hyperkalemia"], "ERC/AHA (resuscitation guidance)",
       "Management of Life-Threatening Hyperkalaemia", 2021,
       "Emergency Management",
       "For hyperkalaemia with ECG changes, give intravenous calcium to stabilise the "
       "myocardium first, then shift potassium intracellularly with insulin plus dextrose "
       "and/or nebulised salbutamol, then remove potassium via binders or dialysis.",
       "expert consensus / resuscitation guidance",
       "https://www.resus.org.uk/library/2021-resuscitation-guidelines",
       ["calcium", "insulin", "dextrose", "salbutamol", "binder", "dialysis", "ecg"]),

    # ── ARDS ─────────────────────────────────────────────────────────────
    _g("ATS2017-ARDS-VT", ["ards"], "ATS/ESICM/SCCM",
       "Mechanical Ventilation in Adult Patients with ARDS", 2017,
       "Tidal Volume and Plateau Pressure",
       "Use low tidal volume ventilation (approximately 4–8 mL/kg predicted body weight) with "
       "plateau pressure limitation in patients with ARDS, as this reduces mortality compared "
       "with higher tidal volumes.",
       "strong recommendation, moderate-high confidence",
       "https://www.atsjournals.org/doi/10.1164/rccm.201703-0548ST",
       ["tidal volume", "plateau pressure", "lung protective", "ventilation"]),
    _g("ATS2017-ARDS-PRONE", ["ards"], "ATS/ESICM/SCCM",
       "Mechanical Ventilation in Adult Patients with ARDS", 2017,
       "Prone Positioning",
       "Prone positioning for more than 12 hours per day is recommended in severe ARDS, "
       "having demonstrated a mortality benefit in this population.",
       "strong recommendation, moderate-high confidence",
       "https://www.atsjournals.org/doi/10.1164/rccm.201703-0548ST",
       ["prone", "positioning", "severe ards", "oxygenation"]),

    # ── stroke ───────────────────────────────────────────────────────────
    _g("AHA2019-STROKE-IVT", ["stroke"], "AHA/ASA",
       "Guidelines for the Early Management of Acute Ischemic Stroke", 2019,
       "Intravenous Thrombolysis",
       "Intravenous alteplase is recommended for eligible patients within 3 hours of symptom "
       "onset, and within 4.5 hours for selected patients. Benefit is strongly time-dependent, "
       "so treatment should not be delayed for ancillary testing.",
       "Class 1, Level of Evidence A",
       "https://www.ahajournals.org/doi/10.1161/STR.0000000000000211",
       ["alteplase", "thrombolysis", "time window", "onset"]),
    _g("AHA2019-STROKE-BP", ["stroke", "hypertensive_emergency"], "AHA/ASA",
       "Guidelines for the Early Management of Acute Ischemic Stroke", 2019,
       "Blood Pressure Management",
       "Blood pressure should be lowered to below 185/110 mmHg before thrombolysis and "
       "maintained below 180/105 mmHg for the first 24 hours afterwards.",
       "Class 1, Level of Evidence B-NR",
       "https://www.ahajournals.org/doi/10.1161/STR.0000000000000211",
       ["blood pressure", "nicardipine", "labetalol", "thrombolysis"]),

    # ── GI bleed ─────────────────────────────────────────────────────────
    _g("ACG2021-UGIB", ["gi_bleed"], "ACG",
       "Guideline on Upper Gastrointestinal and Ulcer Bleeding", 2021,
       "Resuscitation and Endoscopy",
       "Use a restrictive red cell transfusion strategy with a haemoglobin threshold of "
       "approximately 7 g/dL in haemodynamically stable patients, and perform upper endoscopy "
       "within 24 hours of presentation.",
       "conditional recommendation, moderate quality evidence",
       "https://journals.lww.com/ajg/fulltext/2021/05000/acg_clinical_guideline.11.aspx",
       ["transfusion", "haemoglobin", "endoscopy", "restrictive"]),
    _g("AASLD-VARICEAL", ["gi_bleed", "liver_failure"], "AASLD",
       "Portal Hypertensive Bleeding in Cirrhosis", 2016,
       "Acute Variceal Haemorrhage",
       "In suspected variceal bleeding, start a vasoactive agent such as octreotide and "
       "short-course prophylactic antibiotics at presentation, before endoscopic therapy.",
       "Class 1, Level A",
       "https://www.aasld.org/practice-guidelines",
       ["octreotide", "varices", "antibiotic prophylaxis", "cirrhosis"]),

    # ── MI ───────────────────────────────────────────────────────────────
    _g("ACC2021-ACS-REPERFUSION", ["myocardial_infarction"], "ACC/AHA",
       "Coronary Artery Revascularization Guideline", 2021,
       "Reperfusion in STEMI",
       "Primary percutaneous coronary intervention is the preferred reperfusion strategy for "
       "STEMI when it can be delivered promptly by an experienced team; door-to-balloon time "
       "should be minimised.",
       "Class 1, Level of Evidence A",
       "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001038",
       ["pci", "reperfusion", "stemi", "door to balloon"]),

    # ── pancreatitis ─────────────────────────────────────────────────────
    _g("ACG2013-PANC-FLUID", ["pancreatitis"], "ACG",
       "Management of Acute Pancreatitis", 2013,
       "Initial Management",
       "Provide goal-directed intravenous fluid resuscitation in the first 12–24 hours, which "
       "is when it is most beneficial. Routine prophylactic antibiotics are not recommended in "
       "the absence of infected necrosis.",
       "conditional recommendation, moderate quality evidence",
       "https://journals.lww.com/ajg/fulltext/2013/09000/american_college_of_gastroenterology_guideline_.6.aspx",
       ["fluid", "resuscitation", "antibiotic prophylaxis", "necrosis"]),

    # ── pneumonia / COPD ─────────────────────────────────────────────────
    _g("ATS2019-CAP", ["pneumonia"], "ATS/IDSA",
       "Diagnosis and Treatment of Adults with Community-acquired Pneumonia", 2019,
       "Empiric Therapy",
       "Select empiric antibiotic therapy by severity and by risk factors for MRSA and "
       "Pseudomonas, rather than applying broad-spectrum coverage to all patients.",
       "strong recommendation, moderate quality evidence",
       "https://www.atsjournals.org/doi/10.1164/rccm.201908-1581ST",
       ["empiric", "antibiotic", "mrsa", "pseudomonas", "severity"]),
    _g("GOLD-COPD-EXAC", ["copd"], "GOLD",
       "Global Strategy for Diagnosis, Management and Prevention of COPD", 2023,
       "Management of Exacerbations",
       "Treat exacerbations with short-acting bronchodilators, a short course of systemic "
       "corticosteroids, and antibiotics when sputum purulence or ventilatory support "
       "indicates bacterial infection. Use non-invasive ventilation as first-line support in "
       "acute hypercapnic respiratory failure.",
       "evidence category A/B by sub-item",
       "https://goldcopd.org/",
       ["bronchodilator", "corticosteroid", "niv", "hypercapnic"]),

    # ── hepatic encephalopathy ───────────────────────────────────────────
    _g("AASLD-HE", ["liver_failure"], "AASLD/EASL",
       "Hepatic Encephalopathy in Chronic Liver Disease", 2014,
       "Treatment",
       "Lactulose is first-line therapy for overt hepatic encephalopathy, with rifaximin added "
       "as adjunct therapy to reduce recurrence. Identify and treat the precipitating cause.",
       "Class 1, Level A (lactulose); Class 1, Level A (rifaximin add-on)",
       "https://www.aasld.org/practice-guidelines",
       ["lactulose", "rifaximin", "ammonia", "precipitant"]),

    # ── VTE / PE ─────────────────────────────────────────────────────────
    _g("CHEST2021-VTE", ["pulmonary_embolism"], "CHEST",
       "Antithrombotic Therapy for VTE Disease", 2021,
       "Initial Anticoagulation",
       "Start anticoagulation promptly in confirmed pulmonary embolism without high bleeding "
       "risk. Reserve systemic thrombolysis for PE with haemodynamic instability, where the "
       "mortality benefit outweighs the bleeding risk.",
       "strong and conditional recommendations by sub-item",
       "https://journal.chestnet.org/article/S0012-3692(21)01506-3/fulltext",
       ["anticoagulation", "thrombolysis", "haemodynamic instability", "bleeding risk"]),
)

_BY_CONCEPT: Dict[str, List[GuidelineRecord]] = {}
for _r in GUIDELINE_CORPUS:
    for _c in _r.concepts:
        _BY_CONCEPT.setdefault(_c, []).append(_r)


def retrieve_guidelines(
    concepts: Sequence[str],
    query_terms: Optional[Sequence[str]] = None,
    top_k: int = 4,
    require_reviewed: bool = False,
) -> List[dict]:
    """
    Retrieve Level 1 guideline records for the given canonical concept keys.

    Ranking is by concept match first, then overlap with ``query_terms`` against
    each record's keywords and recommendation text. Returns documents in the
    RAG's standard shape; an empty list genuinely means "no guideline on file",
    which is a different state from "retrieval failed".
    """
    seen: set = set()
    candidates: List[GuidelineRecord] = []
    for c in concepts or ():
        for rec in _BY_CONCEPT.get(c, []):
            if rec.doc_id not in seen:
                seen.add(rec.doc_id)
                candidates.append(rec)

    if require_reviewed:
        candidates = [r for r in candidates if r.review_status == "clinician_reviewed"]

    if query_terms:
        qt = {str(t).lower() for t in query_terms if t}

        def score(r: GuidelineRecord) -> int:
            hay = (" ".join(r.keywords) + " " + r.recommendation + " " + r.section).lower()
            return sum(1 for t in qt if t in hay)

        candidates.sort(key=score, reverse=True)

    return [r.to_doc() for r in candidates[:top_k]]


def corpus_stats() -> dict:
    return {
        "version": CORPUS_VERSION,
        "n_records": len(GUIDELINE_CORPUS),
        "n_concepts_covered": len(_BY_CONCEPT),
        "concepts": sorted(_BY_CONCEPT),
        "n_clinician_reviewed": sum(
            1 for r in GUIDELINE_CORPUS if r.review_status == "clinician_reviewed"
        ),
    }
