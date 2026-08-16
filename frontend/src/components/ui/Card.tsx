import type { ReactNode } from "react";
import s from "./Card.module.css";

interface Props {
  title?: string;
  /** Short right-aligned context — a count, a version, a source. */
  note?: ReactNode;
  children: ReactNode;
}

/** The only surface in the app. Everything that looks raised is one of these. */
export function Card({ title, note, children }: Props) {
  return (
    <section className={s.card}>
      {(title || note) && (
        <header className={s.header}>
          {title && <h2 className={s.title}>{title}</h2>}
          {note && <span className={s.note}>{note}</span>}
        </header>
      )}
      <div className={s.stack}>{children}</div>
    </section>
  );
}
