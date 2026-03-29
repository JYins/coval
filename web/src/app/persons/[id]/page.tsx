"use client";

import { useParams } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { AuthGuard } from "@/components/auth-guard";
import {
  askQuestion,
  getBriefing,
  getInteractionSummary,
  getInteractions,
  getPerson,
  rateInteraction,
  uploadConversation,
} from "@/lib/api";
import type {
  AskResult,
  BriefingResult,
  Interaction,
  InteractionSummary,
  Person,
  SessionInfo,
} from "@/lib/types";

function formatDate(value: string | null): string {
  if (!value) {
    return "not set yet";
  }

  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function toIsoValue(value: string): string | undefined {
  if (!value) {
    return undefined;
  }

  return new Date(value).toISOString();
}

async function loadPersonWorkspace(token: string, personId: string) {
  const [nextPerson, nextInteractions, nextSummary] = await Promise.all([
    getPerson(token, personId),
    getInteractions(token, personId),
    getInteractionSummary(token, personId),
  ]);

  return { nextPerson, nextInteractions, nextSummary };
}

function PersonDetailContent({ session }: { session: SessionInfo }) {
  const params = useParams<{ id: string }>();
  const personId = params.id;
  const [person, setPerson] = useState<Person | null>(null);
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [summary, setSummary] = useState<InteractionSummary | null>(null);
  const [briefing, setBriefing] = useState<BriefingResult | null>(null);
  const [answer, setAnswer] = useState<AskResult | null>(null);
  const [question, setQuestion] = useState(
    "What should I remember before I talk to this person again?",
  );
  const [language, setLanguage] = useState("en");
  const [conversationDate, setConversationDate] = useState("");
  const [manualText, setManualText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"manual" | "file_upload">("manual");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isBriefing, setIsBriefing] = useState(false);

  async function loadAll() {
    setError("");

    try {
      const { nextPerson, nextInteractions, nextSummary } =
        await loadPersonWorkspace(session.token, personId);
      setPerson(nextPerson);
      setInteractions(nextInteractions);
      setSummary(nextSummary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not load person");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        const { nextPerson, nextInteractions, nextSummary } =
          await loadPersonWorkspace(session.token, personId);
        if (!cancelled) {
          setPerson(nextPerson);
          setInteractions(nextInteractions);
          setSummary(nextSummary);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "could not load person");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void run();

    return () => {
      cancelled = true;
    };
  }, [personId, session.token]);

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setStatus("");
    setIsUploading(true);

    try {
      await uploadConversation(session.token, {
        personId,
        sourceType: mode,
        language,
        conversationDate: toIsoValue(conversationDate),
        rawContent: mode === "manual" ? manualText : undefined,
        file: mode === "file_upload" ? file : undefined,
      });

      setManualText("");
      setFile(null);
      setStatus("Conversation saved. Personality profile refresh also ran in backend.");
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setStatus("");
    setIsAsking(true);

    try {
      const result = await askQuestion(session.token, {
        person_id: personId,
        question,
        top_k: 5,
      });
      setAnswer(result);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "ask failed");
    } finally {
      setIsAsking(false);
    }
  }

  async function handleBriefing() {
    setError("");
    setStatus("");
    setIsBriefing(true);

    try {
      const result = await getBriefing(session.token, personId, 5);
      setBriefing(result);
      setStatus("Briefing ready.");
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "briefing failed");
    } finally {
      setIsBriefing(false);
    }
  }

  async function handleRate(interactionId: string, userRating: number) {
    setError("");

    try {
      await rateInteraction(session.token, personId, interactionId, userRating);
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "rating failed");
    }
  }

  if (isLoading) {
    return (
      <div className="loading-shell">
        <p className="status-text">Loading person workspace...</p>
      </div>
    );
  }

  if (!person) {
    return (
      <div className="empty-shell">
        <p className="helper-text error-text">{error || "person not found"}</p>
      </div>
    );
  }

  return (
    <AppShell
      email={session.email}
      title={person.name}
      subtitle="Upload fresh context, ask a precise question, then keep the feedback loop honest."
      actions={
        <div className="chips-row">
          <span className="tiny-chip">{person.relationship_type}</span>
          <span className="tiny-chip">last contact {formatDate(person.last_contact)}</span>
        </div>
      }
    >
      {error ? <p className="helper-text error-text">{error}</p> : null}
      {status ? <p className="helper-text">{status}</p> : null}

      <div className="detail-grid">
        <section className="panel-stack">
          <article className="surface-panel">
            <div className="topline">
              <div>
                <p className="mini-label">Conversation input</p>
                <h2 className="step-title">Add new memory</h2>
              </div>
              <div className="pill-switch" role="tablist" aria-label="conversation mode">
                <button
                  type="button"
                  className={mode === "manual" ? "active-pill" : ""}
                  onClick={() => setMode("manual")}
                >
                  Manual
                </button>
                <button
                  type="button"
                  className={mode === "file_upload" ? "active-pill" : ""}
                  onClick={() => setMode("file_upload")}
                >
                  File
                </button>
              </div>
            </div>

            <form className="stack-form" onSubmit={handleUpload}>
              <div className="form-grid">
                <div className="field">
                  <label htmlFor="language">Language</label>
                  <select
                    id="language"
                    name="language"
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                  >
                    <option value="en">English</option>
                    <option value="zh">Chinese</option>
                    <option value="mixed">Mixed</option>
                  </select>
                </div>

                <div className="field">
                  <label htmlFor="conversationDate">Conversation date</label>
                  <input
                    id="conversationDate"
                    name="conversationDate"
                    type="datetime-local"
                    value={conversationDate}
                    onChange={(event) => setConversationDate(event.target.value)}
                  />
                </div>

                {mode === "manual" ? (
                  <div className="field full-span">
                    <label htmlFor="manualText">Raw conversation</label>
                    <textarea
                      id="manualText"
                      name="manualText"
                      value={manualText}
                      onChange={(event) => setManualText(event.target.value)}
                      placeholder="Paste notes, messages, or short meeting memory here..."
                    />
                  </div>
                ) : (
                  <div className="field full-span">
                    <label htmlFor="fileUpload">Chat export</label>
                    <input
                      id="fileUpload"
                      name="fileUpload"
                      type="file"
                      accept=".txt,.csv"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    />
                  </div>
                )}
              </div>

              <button type="submit" className="button" disabled={isUploading}>
                {isUploading ? "Saving..." : "Save conversation"}
              </button>
            </form>
          </article>

          <article className="surface-panel">
            <div className="topline">
              <div>
                <p className="mini-label">Ask</p>
                <h2 className="step-title">Generate advice with current context</h2>
              </div>
              <button
                type="button"
                className="button-secondary"
                onClick={handleBriefing}
                disabled={isBriefing}
              >
                {isBriefing ? "Building..." : "Get briefing"}
              </button>
            </div>

            <form className="stack-form" onSubmit={handleAsk}>
              <div className="field">
                <label htmlFor="question">Question</label>
                <textarea
                  id="question"
                  name="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                />
              </div>
              <button type="submit" className="button" disabled={isAsking}>
                {isAsking ? "Thinking..." : "Ask Coval"}
              </button>
            </form>

            {answer ? (
              <div className="result-panel">
                <div>
                  <p className="mini-label">Latest answer</p>
                  <p className="panel-copy">{answer.answer}</p>
                </div>
                <div className="chunk-list">
                  {answer.retrieved_chunks.map((chunk) => (
                    <article key={`${chunk.chunk_id}-${chunk.rank}`} className="chunk-item">
                      <div className="history-head">
                        <span className="history-type">Chunk #{chunk.rank}</span>
                        <span className="tiny-chip">score {chunk.score.toFixed(3)}</span>
                      </div>
                      <p className="panel-copy">{chunk.chunk_text}</p>
                    </article>
                  ))}
                </div>
              </div>
            ) : null}

            {briefing ? (
              <div className="result-panel">
                <div>
                  <p className="mini-label">Briefing</p>
                  <p className="panel-copy">{briefing.briefing}</p>
                </div>
                <div className="chunk-list">
                  {briefing.retrieved_chunks.map((chunk) => (
                    <article key={`${chunk.chunk_id}-${chunk.rank}`} className="chunk-item">
                      <div className="history-head">
                        <span className="history-type">Chunk #{chunk.rank}</span>
                        <span className="tiny-chip">score {chunk.score.toFixed(3)}</span>
                      </div>
                      <p className="panel-copy">{chunk.chunk_text}</p>
                    </article>
                  ))}
                </div>
              </div>
            ) : null}
          </article>
        </section>

        <aside className="panel-stack">
          <article className="surface-panel">
            <p className="mini-label">Person profile</p>
            <h2 className="step-title">{person.name}</h2>
            <p className="panel-copy">
              {person.notes?.trim() || "No notes yet. Add one or two practical reminders first."}
            </p>
            <div className="chips-row">
              <span className="tiny-chip">first met {formatDate(person.first_met)}</span>
              <span className="tiny-chip">last contact {formatDate(person.last_contact)}</span>
            </div>
          </article>

          {summary ? (
            <article className="surface-panel">
              <div className="topline">
                <div>
                  <p className="mini-label">Feedback summary</p>
                  <h2 className="step-title">How the loop is doing</h2>
                </div>
              </div>
              <div className="summary-grid">
                <div className="summary-tile">
                  <small>Total interactions</small>
                  <p className="summary-value">{summary.total_interactions}</p>
                </div>
                <div className="summary-tile">
                  <small>Rated</small>
                  <p className="summary-value">{summary.rated_interactions}</p>
                </div>
                <div className="summary-tile">
                  <small>Average rating</small>
                  <p className="summary-value">
                    {summary.average_rating !== null ? summary.average_rating.toFixed(1) : "--"}
                  </p>
                </div>
              </div>
              <div className="chips-row">
                {Object.entries(summary.rating_counts).map(([rating, count]) => (
                  <span key={rating} className="tiny-chip">
                    {rating} star {count}
                  </span>
                ))}
              </div>
            </article>
          ) : null}

          <article className="surface-panel">
            <div className="topline">
              <div>
                <p className="mini-label">Interaction history</p>
                <h2 className="step-title">Recent generated outputs</h2>
              </div>
            </div>

            {interactions.length === 0 ? (
              <div className="empty-state">
                No interactions yet. Ask a question or request a briefing first.
              </div>
            ) : (
              <div className="history-list">
                {interactions.map((interaction) => (
                  <article key={interaction.id} className="history-item">
                    <div className="history-head">
                      <span className="history-type">{interaction.interaction_type}</span>
                      <span className="tiny-chip">{formatDate(interaction.created_at)}</span>
                    </div>
                    <p className="panel-copy">{interaction.ai_advice_given}</p>
                    <div className="rating-row">
                      {[1, 2, 3, 4, 5].map((value) => (
                        <button
                          key={value}
                          type="button"
                          className={`rating-button ${
                            interaction.user_rating === value ? "is-active" : ""
                          }`}
                          onClick={() => void handleRate(interaction.id, value)}
                        >
                          {value}
                        </button>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            )}
          </article>
        </aside>
      </div>
    </AppShell>
  );
}

export default function PersonDetailPage() {
  return <AuthGuard>{(session) => <PersonDetailContent session={session} />}</AuthGuard>;
}
