"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState, useTransition } from "react";
import { loginUser, registerUser } from "@/lib/api";
import { saveSession } from "@/lib/auth";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    try {
      await registerUser({ email, password });
      const token = await loginUser({ email, password });
      saveSession({ token: token.access_token, email: email.trim().toLowerCase() });
      startTransition(() => {
        router.push("/dashboard");
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "register failed");
    }
  }

  return (
    <main className="auth-shell">
      <header className="top-nav">
        <Link href="/" className="brand-lockup">
          <span className="brand-text">
            <span className="brand-name">Coval</span>
            <span className="brand-note">relationship memory, but with taste</span>
          </span>
        </Link>
      </header>

      <section className="auth-card">
        <div className="panel-stack">
          <p className="mini-label">Start clean</p>
          <h1 className="auth-title">
            Create the account, then build one useful connection at a time.
          </h1>
          <p className="panel-copy">
            This version stays backend-first on purpose. The goal is not social theater, it is
            continuity you can actually use.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              required
            />
          </div>

          <div className="field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="At least 6 chars"
              required
            />
          </div>

          {error ? <p className="helper-text error-text">{error}</p> : null}

          <button type="submit" className="button" disabled={isPending}>
            {isPending ? "Creating..." : "Create account"}
          </button>
        </form>

        <p className="helper-text">
          Already have one?{" "}
          <Link href="/login" className="mini-label">
            Sign in
          </Link>
        </p>
      </section>
    </main>
  );
}
