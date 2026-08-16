import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { SessionProvider } from "./hooks/SessionContext";
import { CaseScreen } from "./screens/CaseScreen";
import { EvidenceScreen } from "./screens/EvidenceScreen";
import { SystemScreen } from "./screens/SystemScreen";
import { TimelineScreen } from "./screens/TimelineScreen";

/**
 * The provider wraps the shell rather than the Case screen, so navigating to
 * the timeline and back does not discard the conversation.
 */
export const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <SessionProvider>
        <AppShell />
      </SessionProvider>
    ),
    children: [
      { index: true, element: <CaseScreen /> },
      { path: "timeline", element: <TimelineScreen /> },
      { path: "evidence", element: <EvidenceScreen /> },
      { path: "system", element: <SystemScreen /> },
    ],
  },
]);
