"use client";

import Link from "next/link";
import { useDeferredValue, useEffect, useState, useTransition } from "react";
import { AppShell } from "@/components/app-shell";
import { AuthGuard } from "@/components/auth-guard";
import { listPersons } from "@/lib/api";
import type { Person, SessionInfo } from "@/lib/types";

function formatDate(value: string | null): string {
  if (!value) {
    return "not set yet";
  }

  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(value));
}

async function loadPersonList(token: string): Promise<Person[]> {
  return listPersons(token);
}

function DashboardContent({ session }: { session: SessionInfo }) {
  const [persons, setPersons] = useState<Person[]>([]);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  async function loadPeople() {
    setError("");

    try {
      const data = await loadPersonList(session.token);
      setPersons(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not load persons");
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        const data = await loadPersonList(session.token);
        if (!cancelled) {
          setPersons(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "could not load persons");
        }
      }
    }

    void run();

    return () => {
      cancelled = true;
    };
  }, [session.token]);

  const lowered = deferredQuery.trim().toLowerCase();
  const filteredPersons = persons.filter((person) => {
    if (!lowered) {
      return true;
    }

    return (
      person.name.toLowerCase().includes(lowered) ||
      person.relationship_type.toLowerCase().includes(lowered) ||
      (person.notes ?? "").toLowerCase().includes(lowered)
    );
  });

  const relationshipCount = persons.reduce<Record<string, number>>((result, person) => {
    const key = person.relationship_type;
    result[key] = (result[key] ?? 0) + 1;
    return result;
  }, {});

  return (
    <AppShell
      email={session.email}
      title="Relationship dashboard"
      subtitle="Direct, light, and usable. Pick a person, load context, and get ready before the next interaction."
      actions={
        <div className="toolbar">
          <input
            className="search-input"
            type="search"
            placeholder="Search by name or relationship"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button
            type="button"
            className="button-secondary"
            onClick={() => startTransition(() => void loadPeople())}
            disabled={isPending}
          >
            {isPending ? "Refreshing..." : "Refresh"}
          </button>
          <Link href="/persons/new" className="button">
            Add person
          </Link>
        </div>
      }
    >
      <div className="stat-strip">
        <article className="summary-tile">
          <small>Total people</small>
          <p className="summary-value">{persons.length}</p>
        </article>
        <article className="summary-tile">
          <small>Relationship types</small>
          <p className="summary-value">{Object.keys(relationshipCount).length}</p>
        </article>
        <article className="summary-tile">
          <small>Main mix</small>
          <p className="panel-copy">
            {Object.entries(relationshipCount)
              .slice(0, 3)
              .map(([type, count]) => `${type} ${count}`)
              .join(" · ") || "add your first person"}
          </p>
        </article>
      </div>

      <div className="dashboard-grid">
        <section className="surface-panel">
          <div className="topline">
            <div>
              <p className="mini-label">People</p>
              <h2 className="step-title">Current workspace</h2>
            </div>
            <p className="status-text">{filteredPersons.length} visible</p>
          </div>

          {error ? <p className="helper-text error-text">{error}</p> : null}

          {filteredPersons.length === 0 ? (
            <div className="empty-state">
              No person yet. Add one and the rest of the flow starts to make sense very fast.
            </div>
          ) : (
            <div className="person-list">
              {filteredPersons.map((person) => (
                <Link key={person.id} href={`/persons/${person.id}`} className="person-row">
                  <div className="person-row-head">
                    <div>
                      <p className="person-name">{person.name}</p>
                      <div className="person-meta">
                        <span>{person.relationship_type}</span>
                        <span>first met {formatDate(person.first_met)}</span>
                      </div>
                    </div>
                    <span className="tiny-chip">Open</span>
                  </div>
                  <p className="panel-copy">
                    {person.notes?.trim() || "No notes yet. Start with a short context summary."}
                  </p>
                  <div className="tiny-meta">
                    <span>last contact {formatDate(person.last_contact)}</span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>

        <aside className="stack-panel">
          <div className="panel-stack">
            <div>
              <p className="mini-label">Suggested flow</p>
              <h2 className="step-title">Use the product the way it wants to be used.</h2>
            </div>
            <p className="panel-copy">
              1. Create a person. 2. Paste or upload a conversation. 3. Ask for a briefing or a
              next-step suggestion.
            </p>
            <div className="divider" />
            <div>
              <p className="mini-label">Why this page is simple</p>
              <p className="panel-copy">
                Most dashboards over-explain. Here the useful thing is just having one fast place
                to continue the relationship context loop.
              </p>
            </div>
          </div>
        </aside>
      </div>
    </AppShell>
  );
}

export default function DashboardPage() {
  return <AuthGuard>{(session) => <DashboardContent session={session} />}</AuthGuard>;
}
