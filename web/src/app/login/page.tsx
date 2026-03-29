"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useState, useTransition } from "react";
import { loginUser } from "@/lib/api";
import { saveSession } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    try {
      const token = await loginUser({ email, password });
      saveSession({ token: token.access_token, email: email.trim().toLowerCase() });
      startTransition(() => {
        router.push("/dashboard");
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "login failed");
    }
  }

  return (
    <main className="auth-shell">
      <header className="top-nav">
        <Link href="/" className="brand-lockup">
          <span className="brand-text">
            <span className="brand-name">Coval</span>
            <span className="brand-note">shared electrons, shared context</span>
          </span>
        </Link>
      </header>

      <section className="auth-card">
        <div className="panel-stack">
          <p className="mini-label">Welcome back</p>
          <h1 className="auth-title">Sign in and open your relationship workspace.</h1>
          <p className="panel-copy">
            Keep this simple first. Log in, pick a person, upload context, then ask better
            questions.
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
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="At least 6 chars"
              required
            />
          </div>

          {error ? <p className="helper-text error-text">{error}</p> : null}

          <button type="submit" className="button" disabled={isPending}>
            {isPending ? "Opening..." : "Sign in"}
          </button>
        </form>

        <p className="helper-text">
          New here?{" "}
          <Link href="/register" className="mini-label">
            Create an account
          </Link>
        </p>
      </section>
    </main>
  );
}
