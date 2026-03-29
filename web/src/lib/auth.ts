import type { SessionInfo } from "@/lib/types";

const TOKEN_KEY = "coval.token";
const EMAIL_KEY = "coval.email";

export function getStoredSession(): SessionInfo | null {
  if (typeof window === "undefined") {
    return null;
  }

  const token = window.localStorage.getItem(TOKEN_KEY);
  const email = window.localStorage.getItem(EMAIL_KEY);
  if (!token || !email) {
    return null;
  }

  return { token, email };
}

export function saveSession(session: SessionInfo): void {
  window.localStorage.setItem(TOKEN_KEY, session.token);
  window.localStorage.setItem(EMAIL_KEY, session.email);
}

export function clearSession(): void {
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(EMAIL_KEY);
}
