import type { TurnResponse } from "../../api/types";
import { Card } from "../ui/Card";
import { Fact, label } from "../ui/Fact";
import { ModelOutputPanel } from "../ui/ModelOutputPanel";
import { SourceCitation } from "../ui/SourceCitation";
import { Tag } from "../ui/Tag";
import s from "./CasePanel.module.css";

/**
 * The twin as it currently stands.
 *
 * Everything here reads from one turn, so the panel can never show facts from
 * one exchange beside model output from another. There is no patient record
 * behind it — this is the case as stated, which is the honest thing to show.
 */
export function CasePanel({ turn }: { turn: TurnResponse }) {
  const missing = turn.questions.map((q) => q.field);

  return (
    <>
      <Card title="Case" note={`${turn.facts.length} stated`}>
        {turn.facts.length === 0 ? (
          <p className={s.empty}>
            Nothing stated yet. Values appear here as you supply them, each with
            the words it was read from.
          </p>
        ) : (
          <div>
            {turn.facts.map((f) => (
              <Fact key={`${f.field}-${f.turn}`} fact={f} />
            ))}
          </div>
        )}
      </Card>

      {missing.length > 0 && (
        <Card title="Still needed" note="to score this">
          <ul className={s.missing}>
            {turn.questions.map((q) => (
              <li key={q.field}>
                {label(q.field)}
                {q.level === "safety_critical" && <Tag tone="warn">Critical</Tag>}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {turn.predictions && <ModelOutputPanel predictions={turn.predictions} />}

      {turn.sources.length > 0 && (
        <Card title="Evidence" note={`${turn.sources.length} retrieved`}>
          <div>
            {turn.sources.map((src) => (
              <SourceCitation key={src.doc_id} source={src} />
            ))}
          </div>
        </Card>
      )}

      {turn.limitations.length > 0 && (
        <Card title="Limitations">
          <ul className={s.limitations}>
            {turn.limitations.map((l) => (
              <li key={l}>{label(l)}</li>
            ))}
          </ul>
        </Card>
      )}
    </>
  );
}
