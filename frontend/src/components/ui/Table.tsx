import type { ReactNode } from "react";
import s from "./Table.module.css";

export interface Column<T> {
  header: string;
  /** Right-aligned and monospaced. For anything that is a quantity. */
  numeric?: boolean;
  cell: (row: T) => ReactNode;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  empty?: string;
}

/**
 * Generic over the row type, so a screen never casts and never sees `any`.
 *
 * Owns the hairlines, the sticky header and the numeric alignment. A screen
 * that styled its own table would be the first place the spacing rhythm broke.
 */
export function Table<T>({ rows, columns, rowKey, empty }: Props<T>) {
  if (!rows.length) {
    return <p className={s.empty}>{empty ?? "Nothing to show."}</p>;
  }
  return (
    <div className={s.wrap}>
      <table className={s.table}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.header} className={c.numeric ? s.num : undefined}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((c) => (
                <td key={c.header} className={c.numeric ? s.num : undefined}>
                  {c.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
