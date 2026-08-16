import type { Predictions, TaskPrediction } from "../../api/types";
import { Card } from "./Card";
import { Tag } from "./Tag";
import s from "./ModelOutputPanel.module.css";

const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

function Task({ task }: { task: TaskPrediction }) {
  return (
    <div className={s.task}>
      <span className={s.label}>{task.label}</span>
      {task.withheld || task.probability === null ? (
        <>
          <span className={s.dash}>not scored</span>
          <p className={s.withheld}>Withheld: {task.reason}</p>
        </>
      ) : (
        <>
          <span className={s.value}>{pct(task.probability)}</span>
          <div className={s.track}>
            <div
              className={s.fill}
              style={{ width: `${Math.min(task.probability * 100, 100)}%` }}
            />
          </div>
        </>
      )}
    </div>
  );
}

/**
 * What the risk models produced, including what they declined to produce.
 *
 * A withheld task keeps its row. Dropping it would leave the reader assuming
 * the model was never asked, and rendering a zero would read as "no risk" —
 * both are worse than saying the coverage was too thin and why.
 */
export function ModelOutputPanel({ predictions }: { predictions: Predictions }) {
  const { tasks, risk_tier, model_confidence, calibration_statement } = predictions;

  return (
    <Card title="Model estimates" note={predictions.input_kind.replace(/_/g, " ")}>
      <div>
        {tasks.map((t) => (
          <Task key={t.key} task={t} />
        ))}
      </div>

      <div className={s.footer}>
        {risk_tier && <Tag tone="neutral">{risk_tier}</Tag>}
        {model_confidence && <Tag tone="neutral">Confidence: {model_confidence}</Tag>}
      </div>

      {calibration_statement && (
        <p className={s.statement}>{calibration_statement}</p>
      )}
    </Card>
  );
}
