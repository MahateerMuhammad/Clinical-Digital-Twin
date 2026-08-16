import type { ReactNode } from "react";
import s from "./Tag.module.css";

export type Tone = "neutral" | "accent" | "ok" | "warn" | "danger";

/**
 * The only place status colour is applied.
 *
 * Centralised so "which red" is never a per-component decision, and so the
 * reservation of colour for genuinely flagged items is enforceable by reading
 * one file rather than by reviewing every screen.
 */
export function Tag({
  tone = "neutral",
  children,
}: {
  tone?: Tone;
  children: ReactNode;
}) {
  return <span className={`${s.tag} ${s[tone]}`}>{children}</span>;
}
