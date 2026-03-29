"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { clearSession } from "@/lib/auth";
import { LinkedHearts } from "@/components/linked-hearts";

type AppShellProps = {
  email: string;
  title: string;
  subtitle: string;
  actions?: ReactNode;
  children: ReactNode;
};

export function AppShell({
  email,
  title,
  subtitle,
  actions,
  children,
}: AppShellProps) {
  const router = useRouter();

  function handleLogout() {
    clearSession();
    router.push("/login");
  }

  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="brand-lockup">
          <span className="brand-mark">
            <LinkedHearts aria-hidden="true" />
          </span>
          <span className="brand-text">
            <span className="brand-name">Coval</span>
            <span className="brand-note">workspace for warmer continuity</span>
          </span>
        </div>
        <nav className="nav-links">
          <Link href="/dashboard" className="button-ghost">
            Dashboard
          </Link>
          <Link href="/persons/new" className="button-ghost">
            New person
          </Link>
          <button className="button-secondary" onClick={handleLogout} type="button">
            Sign out
          </button>
        </nav>
      </header>

      <section className="workspace-shell">
        <div className="workspace-header">
          <div>
            <p className="mini-label">{email}</p>
            <h1 className="page-title">{title}</h1>
            <p className="panel-copy">{subtitle}</p>
          </div>
          {actions}
        </div>
        {children}
      </section>
    </div>
  );
}
