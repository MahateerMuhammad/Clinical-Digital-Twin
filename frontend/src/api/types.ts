/**
 * Mirrors `backend/schemas.py`. Hand-written rather than generated: the schema
 * is small, and a generator would pull in a build step plus a second source of
 * truth to keep in step with the first.
 *
 * One file, so when the API changes there is exactly one place that must be
 * edited and `tsc` finds every consumer.
 */

/** `answer.py` statuses. A decline is a successful turn, not an error. */
export type TurnStatus =
  | "answered"
  | "capabilities_shown"
  | "declined_incomplete"
  | "declined_no_evidence"
  | "declined_unreviewed"
  | "declined_out_of_scope"
  | "emergency_response";

/** Question urgency, which drives ordering and never colour. */
export type QuestionLevel = "safety_critical" | "required" | "contradiction";

export interface Question {
  field: string;
  text: string;
  level: QuestionLevel;
}

/** A fact the clinician stated, with the span it was read from (spec 13). */
export interface Fact {
  field: string;
  value: string | number | string[] | null;
  quote: string;
  turn: number;
}

/**
 * One model output. `withheld` is not an error state: a task whose payload
 * coverage falls below the retention floor is named and explained rather than
 * scored badly or shown as zero.
 */
export interface TaskPrediction {
  key: string;
  label: string;
  probability: number | null;
  /** Before calibration. Shown only where the difference is instructive. */
  raw_probability: number | null;
  withheld: boolean;
  reason: string;
}

/**
 * The model panel for one turn.
 *
 * Confidence is one label for the whole inference, not a number per task —
 * that is what the runner reports, and a per-task figure would show a
 * precision these models do not have.
 */
export interface Predictions {
  tasks: TaskPrediction[];
  risk_tier: string;
  model_confidence: string;
  calibration_statement: string;
  input_kind: string;
}

/** A retrieved document. `tier` is trust rank, not decoration. */
export interface Source {
  doc_id: string;
  title: string;
  tier: number;
  url: string;
  citation: string;
  review_status: string;
}

export interface TurnResponse {
  session_id: string;
  turn: number;
  status: TurnStatus;
  reply: string;
  questions: Question[];
  citations: string[];
  limitations: string[];
  severity: "none" | "urgent_assess" | "emergency";
  /** null when the turn composed no answer — a question is not verifiable. */
  verified: boolean | null;
  intent: string | null;
  facts: Fact[];
  /** null when the turn ran no model — an empty object would imply it did. */
  predictions: Predictions | null;
  sources: Source[];
  /** Present only when the server runs with CDT_ASSISTANT_DEBUG=1. */
  debug?: TurnDebug;
}

export interface TurnDebug {
  gate: Record<string, unknown> | null;
  extraction: ExtractionRecord | null;
  faithfulness: Record<string, unknown> | null;
  grounding: Record<string, unknown> | null;
}

/** The rejected list is the interesting half: what was proposed and refused. */
export interface ExtractionRecord {
  accepted: { field: string; value: unknown; quote: string }[];
  rejected: { proposal: Record<string, unknown>; reason: string }[];
  used_model: boolean;
  parse_failed: boolean;
}

export interface SessionResponse {
  session_id: string;
  turn: number;
  intent: string | null;
  known_facts: Record<string, string | number | string[] | null>;
  missing: string[];
  messages: { role: string; text: string }[];
}

export interface CorpusStats {
  version: string;
  n_records: number;
  n_concepts_covered: number;
  concepts: string[];
  n_clinician_reviewed: number;
}

export interface Health {
  status: string;
  audience: string;
  models_loaded: boolean;
  models_error: string | null;
  guidelines: CorpusStats;
  patient_corpus: CorpusStats;
  extraction_backend: ExtractionBackend | null;
  debug_mode: boolean;
  sessions: number;
}

export interface ExtractionBackend {
  backend: string;
  available: boolean;
  model: string;
  hosted: boolean;
  base_url: string;
  provider: string;
  /** Never the key itself. This object is public to anyone who can reach /api. */
  key_present: boolean;
}

export interface EvidenceDoc {
  doc_id: string;
  title: string;
  tier: number;
  tier_name: string;
  source_name: string;
  url: string;
  topics: string[];
  review_status: string;
  strength: string;
}

export interface EvidenceResponse {
  stats: CorpusStats;
  documents: EvidenceDoc[];
}
