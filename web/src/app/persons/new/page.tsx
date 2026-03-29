"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState, useTransition } from "react";
import { AppShell } from "@/components/app-shell";
import { AuthGuard } from "@/components/auth-guard";
import { createPerson } from "@/lib/api";
import type { SessionInfo } from "@/lib/types";

function toIsoValue(value: string): string | undefined {
  if (!value) {
    return undefined;
  }

  return new Date(value).toISOString();
}

function NewPersonContent({ session }: { session: SessionInfo }) {
  const router = useRouter();
  const [name, setName] = useState("");
  const [relationshipType, setRelationshipType] = useState("friend");
  const [notes, setNotes] = useState("");
  const [firstMet, setFirstMet] = useState("");
  const [lastContact, setLastContact] = useState("");
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    try {
      const person = await createPerson(session.token, {
        name,
        relationship_type: relationshipType,
        notes,
        first_met: toIsoValue(firstMet),
        last_contact: toIsoValue(lastContact),
      });

      startTransition(() => {
        router.push(`/persons/${person.id}`);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not create person");
    }
  }

  return (
    <AppShell
      email={session.email}
      title="Add a person"
      subtitle="Keep it honest and practical. Name, relationship, a little context, then keep moving."
    >
      <section className="surface-panel">
        <form className="stack-form" onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="field">
              <label htmlFor="name">Name</label>
              <input
                id="name"
                name="name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Jamie Chen"
                required
              />
            </div>

            <div className="field">
              <label htmlFor="relationshipType">Relationship type</label>
              <select
                id="relationshipType"
                name="relationshipType"
                value={relationshipType}
                onChange={(event) => setRelationshipType(event.target.value)}
              >
                <option value="friend">friend</option>
                <option value="client">client</option>
                <option value="romantic">romantic</option>
                <option value="family">family</option>
              </select>
            </div>

            <div className="field">
              <label htmlFor="firstMet">First met</label>
              <input
                id="firstMet"
                name="firstMet"
                type="datetime-local"
                value={firstMet}
                onChange={(event) => setFirstMet(event.target.value)}
              />
            </div>

            <div className="field">
              <label htmlFor="lastContact">Last contact</label>
              <input
                id="lastContact"
                name="lastContact"
                type="datetime-local"
                value={lastContact}
                onChange={(event) => setLastContact(event.target.value)}
              />
            </div>

            <div className="field full-span">
              <label htmlFor="notes">Notes</label>
              <textarea
                id="notes"
                name="notes"
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                placeholder="Where you met, what matters, small sensitivities, topics worth remembering..."
              />
            </div>
          </div>

          {error ? <p className="helper-text error-text">{error}</p> : null}

          <div className="inline-actions">
            <button type="submit" className="button" disabled={isPending}>
              {isPending ? "Creating..." : "Create person"}
            </button>
            <button
              type="button"
              className="button-ghost"
              onClick={() => router.push("/dashboard")}
            >
              Back to dashboard
            </button>
          </div>
        </form>
      </section>
    </AppShell>
  );
}

export default function NewPersonPage() {
  return <AuthGuard>{(session) => <NewPersonContent session={session} />}</AuthGuard>;
}
