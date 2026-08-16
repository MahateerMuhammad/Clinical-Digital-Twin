import { Link } from "react-router-dom";
import type { Exchange } from "../hooks/useSession";
import { Card } from "../components/ui/Card";
import { label } from "../components/ui/Fact";
import { PageHeader } from "../components/ui/PageHeader";
import { Tag, type Tone } from "../components/ui/Tag";
import { useSessionContext } from "../hooks/SessionContext";
import s from "./TimelineScreen.module.css";

const OUTCOME: Record<string, { tone: Tone; text: string }> = {
  answered: { tone: "ok", text: "Answered" },
  capabilities_shown: { tone: "neutral", text: "Session opened" },
  declined_incomplete: { tone: "neutral", text: "Asked for more" },
  declined_no_evidence: { tone: "warn", text: "No source on file" },
  declined_unreviewed: { tone: "warn", text: "Source not reviewed" },
  declined_out_of_scope: { tone: "neutral", text: "Outside scope" },
  emergency_response: { tone: "danger", text: "Urgent" },
};

function Turn({ exchange, index }: { exchange: Exchange; index: number }) {
  const { question, turn } = exchange;
  const outcome = OUTCOME[turn.status];
  const newFacts = turn.facts.filter((f) => f.turn === turn.turn);
  const rejected = turn.debug?.extraction?.rejected ?? [];

  return (
    <li className={s.turn}>
      <div className={s.marker}>
        <span className={s.index}>{index}</span>
      </div>

      <Card
        note={
          <>
            {turn.intent && <Tag tone="neutral">{turn.intent.replace(/_/g, " ")}</Tag>}
            {outcome && <Tag tone={outcome.tone}>{outcome.text}</Tag>}
          </>
        }
      >
        {question ? (
          <p className={s.asked}>{question}</p>
        ) : (
          <p className={s.faint}>Session opened with the capability message.</p>
        )}

        <dl className={s.detail}>
          {newFacts.length > 0 && (
            <div>
              <dt>Read from this message</dt>
              <dd>
                {newFacts.map((f) => (
                  <span key={f.field} className={s.chip}>
                    {label(f.field)}
                    <span className="num">{String(f.value)}</span>
                  </span>
                ))}
              </dd>
            </div>
          )}

          {turn.questions.length > 0 && (
            <div>
              <dt>Asked for</dt>
              <dd className={s.plain}>
                {turn.questions.map((q) => label(q.field)).join(", ")}
              </dd>
            </div>
          )}

          {turn.sources.length > 0 && (
            <div>
              <dt>Retrieved</dt>
              <dd className={s.plain}>
                {turn.sources.length} document{turn.sources.length > 1 ? "s" : ""},
                tier {Math.min(...turn.sources.map((x) => x.tier))}
              </dd>
            </div>
          )}

          {turn.predictions && (
            <div>
              <dt>Models run</dt>
              <dd className={s.plain}>
                {turn.predictions.tasks.filter((t) => !t.withheld).length} scored,{" "}
                {turn.predictions.tasks.filter((t) => t.withheld).length} withheld
              </dd>
            </div>
          )}

          <div>
            <dt>Grounding</dt>
            <dd className={s.plain}>
              {turn.verified === null
                ? "Not applicable, nothing was asserted"
                : turn.verified
                  ? "Every claim traced to a supplied value or a retrieved document"
                  : "Failed, output withheld"}
            </dd>
          </div>

          {rejected.length > 0 && (
            <div>
              <dt>Refused at extraction</dt>
              <dd className={s.plain}>
                {rejected.map((r, i) => (
                  <span key={i} className={s.rejected}>
                    {r.reason}
                  </span>
                ))}
              </dd>
            </div>
          )}
        </dl>
      </Card>
    </li>
  );
}

export function TimelineScreen() {
  const { exchanges } = useSessionContext();
  const debugOn = exchanges.some((e) => e.turn.debug);

  return (
    <>
      <PageHeader
        title="Session timeline"
        subtitle="What happened on each turn: what was read, what was asked for,
                  what was retrieved, and whether the answer was verified."
      />

      {exchanges.length <= 1 ? (
        <Card>
          <p className={s.faint}>
            Nothing yet. Work through a case on the{" "}
            <Link to="/">Case screen</Link> and each turn is traced here.
          </p>
        </Card>
      ) : (
        <ol className={s.list}>
          {exchanges.map((exchange, i) => (
            <Turn key={i} exchange={exchange} index={i} />
          ))}
        </ol>
      )}

      {!debugOn && exchanges.length > 1 && (
        <p className={s.note}>
          Extraction proposals that were refused are recorded but not returned.
          Start the backend with <code>CDT_ASSISTANT_DEBUG=1</code> to include
          them, along with the gate decision and the grounding report.
        </p>
      )}
    </>
  );
}
