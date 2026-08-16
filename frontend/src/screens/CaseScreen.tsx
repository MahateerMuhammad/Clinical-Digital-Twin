import { useState } from "react";
import { CasePanel } from "../components/case/CasePanel";
import { Card } from "../components/ui/Card";
import { Markdown } from "../components/ui/Markdown";
import { PageHeader } from "../components/ui/PageHeader";
import { Tag, type Tone } from "../components/ui/Tag";
import { useSessionContext } from "../hooks/SessionContext";
import s from "./CaseScreen.module.css";

/** Openers that exercise a different path each: evidence, models, boundary. */
const EXAMPLES = [
  "What is the first-line vasopressor in septic shock?",
  "72F septic shock, Cr 3.2, BUN 48, WBC 19.4, bicarb 14, Na 131, K 5.2, plt 88, hct 29, glucose 210, SBP 82, HR 118. What is her mortality risk?",
  "What's the diagnosis?",
];

/** A status is a fact about the turn, not an alarm. Only an outright refusal
 *  for want of evidence earns a warm colour; declines are informational. */
const STATUS: Record<string, { tone: Tone; text: string }> = {
  answered: { tone: "ok", text: "Answered" },
  capabilities_shown: { tone: "neutral", text: "Ready" },
  declined_incomplete: { tone: "neutral", text: "Needs more information" },
  declined_no_evidence: { tone: "warn", text: "No source on file" },
  declined_unreviewed: { tone: "warn", text: "Source not reviewed" },
  declined_out_of_scope: { tone: "neutral", text: "Outside scope" },
  emergency_response: { tone: "danger", text: "Urgent" },
};

export function CaseScreen() {
  const { exchanges, latest, pending, error, send, reset } = useSessionContext();
  const [draft, setDraft] = useState("");

  const submit = () => {
    void send(draft);
    setDraft("");
  };

  return (
    <div className={s.screen}>
      <div>
        <PageHeader
          title="Case"
          subtitle="Everything the twin knows comes from this conversation.
                    There is no record behind it."
          action={
            <button className={s.reset} onClick={() => void reset()}>
              Clear case
            </button>
          }
        />

        <div className={s.thread}>
          {exchanges.map(({ question, turn }, i) => {
            const status = STATUS[turn.status];
            return (
              <div key={i} className={s.thread}>
                {question && <p className={s.asked}>{question}</p>}
                <Card
                  note={
                    <>
                      {status && <Tag tone={status.tone}>{status.text}</Tag>}
                      {turn.verified && <Tag tone="ok">Grounding verified</Tag>}
                    </>
                  }
                >
                  <Markdown text={turn.reply} />
                </Card>
              </div>
            );
          })}

          {pending && (
            <p className={s.waiting}>
              Reading the message, checking what is missing, and retrieving
              evidence… this can take a moment on a topic not seen before.
            </p>
          )}
          {error && <p className={s.error}>{error}</p>}
        </div>

        <div className={s.composer}>
          <div className={s.field}>
            <textarea
              className={s.input}
              rows={2}
              value={draft}
              placeholder="Describe the case, or ask a guideline question…"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            <button
              className={s.send}
              onClick={submit}
              disabled={pending || !draft.trim()}
            >
              Send
            </button>
          </div>

          {exchanges.length <= 1 && (
            <div className={s.examples}>
              {EXAMPLES.map((e) => (
                <button key={e} className={s.example} onClick={() => void send(e)}>
                  {e.length > 60 ? `${e.slice(0, 60)}…` : e}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <aside className={s.side}>{latest && <CasePanel turn={latest} />}</aside>
    </div>
  );
}
