import type { Source } from "../../api/types";
import { Tag } from "./Tag";
import s from "./SourceCitation.module.css";

/**
 * One retrieved document.
 *
 * Review status is shown, not hidden. The corpus is entirely unreviewed, and a
 * citation that looks authoritative while nobody clinical has read it is the
 * more dangerous of the two presentations.
 */
export function SourceCitation({ source }: { source: Source }) {
  return (
    <article className={s.source}>
      <p className={s.title}>{source.title}</p>
      <div className={s.meta}>
        <Tag tone="neutral">Tier {source.tier}</Tag>
        <Tag tone={source.review_status === "unreviewed" ? "warn" : "ok"}>
          {source.review_status === "unreviewed"
            ? "Not clinician-reviewed"
            : "Clinician-reviewed"}
        </Tag>
        {source.url && (
          <a className={s.link} href={source.url} target="_blank" rel="noreferrer">
            Source ↗
          </a>
        )}
      </div>
    </article>
  );
}
