import type { ReactNode } from "react";
import s from "./PageHeader.module.css";

/** Screen chrome, defined once so four screens cannot drift apart. */
export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <header className={s.head}>
      <div>
        <h1 className={s.title}>{title}</h1>
        {subtitle && <p className={s.subtitle}>{subtitle}</p>}
      </div>
      {action}
    </header>
  );
}
