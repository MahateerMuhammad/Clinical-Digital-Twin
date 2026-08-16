import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { EvidenceDoc } from "../api/types";
import { Card } from "../components/ui/Card";
import { PageHeader } from "../components/ui/PageHeader";
import { Table, type Column } from "../components/ui/Table";
import { Tag } from "../components/ui/Tag";
import { useAsync } from "../hooks/useAsync";
import s from "./EvidenceScreen.module.css";

const readable = (concept: string) =>
  concept.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export function EvidenceScreen() {
  const { data, error, loading } = useAsync(api.evidence);
  const [concept, setConcept] = useState<string | null>(null);

  const documents = useMemo(
    () =>
      concept
        ? (data?.documents ?? []).filter((d) => d.topics.includes(concept))
        : (data?.documents ?? []),
    [data, concept],
  );

  const columns: Column<EvidenceDoc>[] = [
    { header: "Document", cell: (d) => <span className={s.title}>{d.title}</span> },
    { header: "Source", cell: (d) => d.source_name },
    {
      header: "Covers",
      cell: (d) => (
        <span className={s.topics}>{d.topics.map(readable).join(", ")}</span>
      ),
    },
    { header: "Strength", cell: (d) => <span className={s.faint}>{d.strength}</span> },
    {
      header: "Review",
      cell: (d) => (
        <Tag tone={d.review_status === "unreviewed" ? "warn" : "ok"}>
          {d.review_status === "unreviewed" ? "Unreviewed" : "Reviewed"}
        </Tag>
      ),
    },
  ];

  if (loading) return <p className={s.faint}>Loading the corpus…</p>;
  if (error) return <p className={s.faint}>{error}</p>;
  if (!data) return null;

  const { stats } = data;

  return (
    <>
      <PageHeader
        title="Evidence"
        subtitle="Everything the assistant is able to cite. A question outside these
                  concepts is declined rather than answered from memory."
      />

      <div className={s.stack}>
        <Card title="Corpus" note={`version ${stats.version}`}>
          <div className={s.stats}>
            <Stat value={stats.n_records} label="records" />
            <Stat value={stats.n_concepts_covered} label="concepts covered" />
            <Stat
              value={stats.n_clinician_reviewed}
              label="clinician-reviewed"
              flagged={stats.n_clinician_reviewed === 0}
            />
          </div>
          {stats.n_clinician_reviewed === 0 && (
            <p className={s.caveat}>
              No document here has been reviewed by a clinician. Retrieval proves
              a claim traces to a source; it does not prove the source is right,
              or that it applies to the patient in front of you.
            </p>
          )}
        </Card>

        <Card title="Concepts" note={concept ? "filtered" : "all"}>
          <div className={s.concepts}>
            <button
              className={concept === null ? s.chipOn : s.chip}
              onClick={() => setConcept(null)}
            >
              All
            </button>
            {stats.concepts.map((c) => (
              <button
                key={c}
                className={c === concept ? s.chipOn : s.chip}
                onClick={() => setConcept(c === concept ? null : c)}
              >
                {readable(c)}
              </button>
            ))}
          </div>
        </Card>

        <Card
          title="Documents"
          note={`${documents.length} of ${stats.n_records}`}
        >
          <Table
            rows={documents}
            columns={columns}
            rowKey={(d) => d.doc_id}
            empty="No document covers this concept."
          />
        </Card>
      </div>
    </>
  );
}

function Stat({
  value,
  label,
  flagged,
}: {
  value: number;
  label: string;
  flagged?: boolean;
}) {
  return (
    <div>
      <p className={flagged ? s.figureFlagged : s.figure}>{value}</p>
      <p className={s.figureLabel}>{label}</p>
    </div>
  );
}
