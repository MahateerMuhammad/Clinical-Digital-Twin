import { api } from "../api/client";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { Tag } from "../components/ui/Tag";
import { useAsync } from "../hooks/useAsync";
import s from "./SystemScreen.module.css";

/** What this system will not do. Stated rather than discovered by trying. */
const BOUNDARIES = [
  "It has no connection to an EMR and cannot open a chart or look up a patient. Every value comes from what is typed into a case.",
  "It does not diagnose. Assigning a diagnosis is a clinical judgement held deliberately outside its scope.",
  "The language model reads values out of your text and, optionally, rewords a finished answer. It never produces a number and never decides whether there is enough information to answer.",
  "Risk estimates come from models trained on MIMIC-IV and carry that cohort's biases. They describe patients like those, not this one.",
  "Retrieval proves a claim traces to a source. It does not prove the source is correct or that it applies here.",
];

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className={s.row}>
      <span className={s.label}>{label}</span>
      <span className={s.value}>{children}</span>
    </div>
  );
}

export function SystemScreen() {
  const { data, error, loading } = useAsync(api.health);

  if (loading) return <p className={s.faint}>Checking the backend…</p>;
  if (error) {
    return (
      <>
        <PageHeader title="System" />
        <Card title="Backend unreachable">
          <p className={s.faint}>{error}</p>
          <p className={s.faint}>
            Start it with
            <code> PYTHONPATH=. .venv/bin/uvicorn backend.main:app --port 8010</code>
          </p>
        </Card>
      </>
    );
  }
  if (!data) return null;

  const backend = data.extraction_backend;

  return (
    <>
      <PageHeader
        title="System"
        subtitle="What is loaded, what it is running on, and what it will not do."
      />

      <div className={s.grid}>
        <Card title="Risk models">
          <Row label="Loaded">
            <Tag tone={data.models_loaded ? "ok" : "warn"}>
              {data.models_loaded ? "Ready" : "Not loaded"}
            </Tag>
          </Row>
          {data.models_error && <Row label="Error">{data.models_error}</Row>}
          <Row label="Audience">{data.audience}</Row>
          <Row label="Open sessions">
            <span className="num">{data.sessions}</span>
          </Row>
        </Card>

        <Card title="Language model" note="extraction and rewording only">
          {backend ? (
            <>
              <Row label="Model">
                <span className="num">{backend.model}</span>
              </Row>
              <Row label="Provider">{backend.provider}</Row>
              <Row label="Key">
                <Tag tone={backend.key_present ? "ok" : "warn"}>
                  {backend.key_present ? "Present" : "Absent"}
                </Tag>
              </Row>
            </>
          ) : (
            <p className={s.faint}>
              No language model configured. Extraction runs on deterministic
              patterns, which read a full presentation line but not unusual
              phrasing. Nothing else changes: the gate, the risk models,
              retrieval and every safety check run without one.
            </p>
          )}
        </Card>

        <Card title="Guideline corpus" note={`version ${data.guidelines.version}`}>
          <Row label="Records">
            <span className="num">{data.guidelines.n_records}</span>
          </Row>
          <Row label="Concepts covered">
            <span className="num">{data.guidelines.n_concepts_covered}</span>
          </Row>
          <Row label="Clinician-reviewed">
            <Tag tone={data.guidelines.n_clinician_reviewed ? "ok" : "warn"}>
              {data.guidelines.n_clinician_reviewed} of {data.guidelines.n_records}
            </Tag>
          </Row>
        </Card>

        <Card title="Deployment">
          <Row label="Authentication">
            <Tag tone="warn">None</Tag>
          </Row>
          <Row label="Binding">localhost only</Row>
          <Row label="Debug trace">{data.debug_mode ? "On" : "Off"}</Row>
          <p className={s.faint}>
            Not to receive real patient data. This instance runs over
            de-identified, published MIMIC-IV.
          </p>
        </Card>
      </div>

      <Card title="What this system will not do">
        <ul className={s.boundaries}>
          {BOUNDARIES.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
      </Card>
    </>
  );
}
