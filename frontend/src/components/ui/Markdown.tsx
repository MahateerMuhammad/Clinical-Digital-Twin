import type { JSX } from "react";
import s from "./Markdown.module.css";

/**
 * The composed reply, rendered.
 *
 * Deliberately not a markdown library. The server composes this text itself and
 * emits a known, narrow subset — headings, bullets, tables, bold, code — so a
 * general parser would add a dependency and an XSS surface to handle syntax
 * that never arrives. `dangerouslySetInnerHTML` is never used here: every node
 * below is constructed, so nothing in a reply can inject markup.
 */
export function Markdown({ text }: { text: string }) {
  const blocks: JSX.Element[] = [];
  const lines = text.split("\n");

  for (let i = 0; i < lines.length; i += 1) {
    const line = (lines[i] ?? "").trimEnd();

    if (line.startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && (lines[i] ?? "").startsWith("|")) {
        rows.push(
          (lines[i] ?? "")
            .trim()
            .replace(/^\||\|$/g, "")
            .split("|")
            .map((c) => c.trim()),
        );
        i += 1;
      }
      i -= 1;
      const [head, , ...body] = rows;
      blocks.push(
        <table key={blocks.length} className={s.table}>
          <thead>
            <tr>{head?.map((c, n) => <th key={n}>{inline(c)}</th>)}</tr>
          </thead>
          <tbody>
            {body.map((row, n) => (
              <tr key={n}>{row.map((c, m) => <td key={m}>{inline(c)}</td>)}</tr>
            ))}
          </tbody>
        </table>,
      );
      continue;
    }

    if (!line.trim() || line.trim() === "---") continue;

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1]?.length ?? 1;
      const Tag = (level <= 2 ? "h3" : "h4") as "h3" | "h4";
      blocks.push(
        <Tag key={blocks.length} className={s.heading}>
          {inline(heading[2] ?? "")}
        </Tag>,
      );
      continue;
    }

    if (/^[-*]\s+/.test(line.trim())) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i] ?? "")) {
        items.push((lines[i] ?? "").trim().replace(/^[-*]\s+/, ""));
        i += 1;
      }
      i -= 1;
      blocks.push(
        <ul key={blocks.length} className={s.list}>
          {items.map((item, n) => <li key={n}>{inline(item)}</li>)}
        </ul>,
      );
      continue;
    }

    blocks.push(
      <p key={blocks.length} className={s.para}>
        {inline(line)}
      </p>,
    );
  }

  return <div className={s.body}>{blocks}</div>;
}

/** Bold, code and bracketed citations. Everything else is literal text. */
function inline(text: string): (string | JSX.Element)[] {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\])/g)
    .filter(Boolean)
    .map((part, n) => {
      if (part.startsWith("**")) return <strong key={n}>{part.slice(2, -2)}</strong>;
      if (part.startsWith("`")) return <code key={n}>{part.slice(1, -1)}</code>;
      if (part.startsWith("[")) return <cite key={n} className={s.cite}>{part}</cite>;
      return part;
    });
}
