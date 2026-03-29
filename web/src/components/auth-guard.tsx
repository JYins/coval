"use client";

import { useRouter } from "next/navigation";
import { useEffect, useSyncExternalStore, type ReactNode } from "react";
import { getStoredSession } from "@/lib/auth";
import type { SessionInfo } from "@/lib/types";

type AuthGuardProps = {
  children: (session: SessionInfo) => ReactNode;
};

function subscribe() {
  return () => undefined;
}

function getSessionSnapshot(): string | null {
  const session = getStoredSession();
  return session ? JSON.stringify(session) : null;
}

export function AuthGuard({ children }: AuthGuardProps) {
  const router = useRouter();
  const sessionValue = useSyncExternalStore(
    subscribe,
    getSessionSnapshot,
    () => null,
  );
  const session = sessionValue ? (JSON.parse(sessionValue) as SessionInfo) : null;

  useEffect(() => {
    if (!session) {
      router.replace("/login");
    }
  }, [router, session]);

  if (!session) {
    return (
      <div className="loading-shell">
        <p className="status-text">Checking your session...</p>
      </div>
    );
  }

  return <>{children(session)}</>;
}
