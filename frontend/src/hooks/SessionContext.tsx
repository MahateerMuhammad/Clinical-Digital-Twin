import { createContext, useContext, type ReactNode } from "react";
import { useSession } from "./useSession";

type SessionValue = ReturnType<typeof useSession>;

const Ctx = createContext<SessionValue | null>(null);

/**
 * One conversation, shared by every screen that reads it.
 *
 * The Case screen drives it and the timeline reads it. Holding the state inside
 * Case and refetching in the timeline would give two screens two ideas of the
 * same session, and the second would silently open its own.
 */
export function SessionProvider({ children }: { children: ReactNode }) {
  return <Ctx.Provider value={useSession()}>{children}</Ctx.Provider>;
}

export function useSessionContext(): SessionValue {
  const value = useContext(Ctx);
  if (!value) throw new Error("useSessionContext used outside SessionProvider");
  return value;
}
