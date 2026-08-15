import { api } from "../api/client";
import { useAsync } from "../hooks/useAsync";

/** Placeholder: proves the dev proxy reaches the backend. Rebuilt on the
 *  primitives once they exist. */
export function SystemScreen() {
  const { data, error, loading } = useAsync(api.health);

  if (loading) return <p>Checking the backend…</p>;
  if (error) return <p>{error}</p>;
  if (!data) return null;

  return (
    <>
      <h1>System</h1>
      <p>
        Models loaded: <span className="num">{String(data.models_loaded)}</span>
        {" · "}
        Guidelines: <span className="num">{data.guidelines.n_records}</span>{" "}
        records, <span className="num">{data.guidelines.n_concepts_covered}</span>{" "}
        concepts
        {" · "}
        Backend: {data.extraction_backend?.model ?? "deterministic only"}
      </p>
    </>
  );
}
