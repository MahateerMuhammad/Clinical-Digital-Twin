import type { Driver } from "../../api/types";
import s from "./DriverBars.module.css";

/**
 * What moved the mortality estimate, as signed bars about a zero line.
 *
 * Hand-drawn rather than charted: three CSS rules and a percentage width do
 * everything a library would, without 500KB and a set of defaults to override.
 *
 * Bars are scaled against the largest absolute contribution in the set, so the
 * chart shows *relative* influence. It deliberately does not scale against a
 * fixed range: SHAP values are log-odds, and a reader comparing bar lengths
 * across two different patients would be comparing two different scales.
 */
export function DriverBars({ drivers }: { drivers: Driver[] }) {
  if (!drivers.length) return null;

  const max = Math.max(...drivers.map((d) => Math.abs(d.contribution)));
  const anyMissing = drivers.some((d) => !d.supplied);

  return (
    <div>
      {drivers.map((d) => {
        const width = max ? (Math.abs(d.contribution) / max) * 50 : 0;
        const positive = d.contribution >= 0;
        return (
          <div key={d.feature} className={s.row}>
            <span className={d.supplied ? s.label : `${s.label} ${s.labelMissing}`}>
              {d.label}
              {!d.supplied && " (not supplied)"}
            </span>
            <span className={s.value}>
              {d.contribution > 0 ? "+" : ""}
              {d.contribution.toFixed(3)}
            </span>
            <div className={s.track}>
              <span className={s.axis} />
              <span
                className={d.supplied ? s.bar : `${s.bar} ${s.barMissing}`}
                style={
                  positive
                    ? { left: "50%", width: `${width}%` }
                    : { right: "50%", width: `${width}%` }
                }
              />
            </div>
          </div>
        );
      })}

      <p className={s.note}>
        SHAP contributions in log-odds for the in-hospital mortality model. Right
        of the line raises the estimate, left lowers it. These describe how the
        model weighed this input; they are not clinical causes.
        {anyMissing && (
          <>
            {" "}
            <span className={`${s.legend} ${s.barMissing}`} />
            marks a value you did not supply: the model is responding to its
            absence, not to the patient.
          </>
        )}
      </p>
    </div>
  );
}
