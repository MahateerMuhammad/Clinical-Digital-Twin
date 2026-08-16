import { NavLink, Outlet } from "react-router-dom";
import s from "./AppShell.module.css";

const NAV = [
  { to: "/", label: "Case", end: true },
  { to: "/timeline", label: "Session timeline" },
  { to: "/evidence", label: "Evidence" },
  { to: "/system", label: "System" },
] as const;

/** Page chrome, defined once. Screens render into the outlet and own nothing
 *  above their own heading. */
export function AppShell() {
  return (
    <div className={s.shell}>
      <aside className={s.sidebar}>
        <div className={s.brand}>
          <span className={s.brandName}>Clinical Digital Twin</span>
          <span className={s.brandNote}>Decision support · MIMIC-IV</span>
        </div>

        <nav className={s.nav}>
          {NAV.map(({ to, label, ...rest }) => (
            <NavLink
              key={to}
              to={to}
              {...rest}
              className={({ isActive }) =>
                isActive ? `${s.link} ${s.linkActive}` : s.link
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        <p className={s.notice}>
          Localhost, no authentication. Not for real patient data. This instance
          is a demonstration over de-identified MIMIC-IV.
        </p>
      </aside>

      <main className={s.main}>
        <Outlet />
      </main>
    </div>
  );
}
