import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { SystemScreen } from "./screens/SystemScreen";

/** Screens are added here as they are built; the shell is already shared, so
 *  none of them repeats page chrome. */
export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [{ path: "system", element: <SystemScreen /> }],
  },
]);
