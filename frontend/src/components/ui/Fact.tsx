import type { Fact as FactData } from "../../api/types";
import s from "./Fact.module.css";

/** Field names are the payload contract's; these are the clinician's words. */
const LABELS: Record<string, string> = {
  age: "Age",
  sex: "Sex",
  primary_diagnosis: "Diagnosis",
  creatinine_max: "Creatinine, peak",
  bun_max: "BUN, peak",
  wbc_max: "White cells, peak",
  bicarbonate_min: "Bicarbonate, lowest",
  sodium_min: "Sodium, lowest",
  potassium_max: "Potassium, peak",
  platelets_min: "Platelets, lowest",
  hematocrit_min: "Haematocrit, lowest",
  glucose_max: "Glucose, peak",
  lactate_max: "Lactate, peak",
  sbp_min: "Systolic BP, lowest",
  hr_max: "Heart rate, peak",
  medication_name: "Medication",
  term: "Term",
};

export const label = (field: string) =>
  LABELS[field] ?? field.replace(/_/g, " ");

const format = (v: FactData["value"]) =>
  Array.isArray(v) ? v.join(", ") : v === null ? "not stated" : String(v);

/** One stated value with the span it was read from (spec 13). */
export function Fact({ fact }: { fact: FactData }) {
  return (
    <div className={s.row}>
      <span className={s.label}>{label(fact.field)}</span>
      <span className={s.value}>{format(fact.value)}</span>
      {fact.quote && <span className={s.quote}>{fact.quote}</span>}
    </div>
  );
}
