import Link from "next/link";
import { LinkedHearts } from "@/components/linked-hearts";

const workflow = [
  {
    step: "01",
    title: "Collect the right memory",
    body: "Manual notes, chat uploads, and later voice or OCR all land in one quiet place.",
  },
  {
    step: "02",
    title: "Retrieve only what matters",
    body: "Coval pulls the right fragments first, because context quality matters more than model hype.",
  },
  {
    step: "03",
    title: "Walk into the next meeting ready",
    body: "Briefings, advice, and follow-up prompts stay practical so continuity feels natural.",
  },
];

export default function Home() {
  return (
    <main className="landing-page">
      <div className="site-shell">
        <header className="top-nav">
          <Link href="/" className="brand-lockup">
            <span className="brand-mark">
              <LinkedHearts aria-hidden="true" />
            </span>
            <span className="brand-text">
              <span className="brand-name">Coval</span>
              <span className="brand-note">shared electrons, shared context</span>
            </span>
          </Link>
          <nav className="nav-links">
            <Link href="/login" className="button-ghost">
              Sign in
            </Link>
            <Link href="/register" className="button-secondary">
              Create account
            </Link>
          </nav>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <p className="hero-kicker">Relationship Memory System</p>
            <h1 className="hero-title">Keep the link warm before the next hello.</h1>
            <p className="hero-body">
              Coval is a networking-first relationship memory backend turned into a premium demo
              surface. The name comes from covalent bond: two atoms share electrons, two people
              share context. This product tries to keep that context usable, not creepy.
            </p>
            <div className="hero-actions">
              <Link href="/register" className="button">
                Start the workspace
              </Link>
              <Link href="/dashboard" className="button-secondary">
                Open demo flow
              </Link>
            </div>
            <div className="hero-meta">
              <span className="meta-chip">RAG-backed continuity</span>
              <span className="meta-chip">built from my retrieval eval work</span>
              <span className="meta-chip">briefings, ask, feedback loop</span>
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-disc" />
            <div className="visual-frame">
              <LinkedHearts className="linked-hearts" />
            </div>
            <div className="hero-caption">
              <p className="mini-label">Visual thesis</p>
              <p>
                Quiet yellow like old paper, one calm teal for signal, one dark ink for trust.
                The connection stays visible before the UI gets in the way.
              </p>
            </div>
          </div>
        </section>

        <section className="landing-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">How It Works</p>
              <h2 className="section-title">Three steps, not ten tabs pretending to help.</h2>
            </div>
            <p className="panel-copy">
              The first version stays direct on purpose. One person, one stream of context, one
              answer surface you can actually use before a coffee chat, a client call, or a warm
              follow-up.
            </p>
          </div>
          <div className="workflow-grid">
            {workflow.map((item) => (
              <article key={item.step} className="workflow-step">
                <span className="step-number">{item.step}</span>
                <h3 className="step-title">{item.title}</h3>
                <p className="panel-copy">{item.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="landing-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Why This Design</p>
              <h2 className="section-title">Premium enough to trust, simple enough to use fast.</h2>
            </div>
            <p className="panel-copy">
              I kept the interface closer to a calm operating surface than a marketing dashboard.
              You see the function first: people, conversations, briefings, answers, ratings.
            </p>
          </div>
          <div className="signal-grid">
            <div className="surface-panel">
              <div className="signal-list">
                <div className="signal-row">
                  <span>Palette</span>
                  <span>paper yellow, qing teal, ink</span>
                </div>
                <div className="signal-row">
                  <span>Interaction</span>
                  <span>big tap targets, light blur, soft motion</span>
                </div>
                <div className="signal-row">
                  <span>Positioning</span>
                  <span>networking and continuity, not dating gimmicks</span>
                </div>
                <div className="signal-row">
                  <span>Feeling</span>
                  <span>worth paying for because it saves social energy</span>
                </div>
              </div>
            </div>
            <div className="signal-strip">
              <p className="mini-label">Coval logic</p>
              <h3 className="step-title">The link is the product.</h3>
              <p className="panel-copy">
                Not memory for memory’s sake. The useful part is what happens right before you
                reach out again: what to remember, how to phrase it, what not to miss.
              </p>
              <div className="inline-actions">
                <Link href="/dashboard" className="button">
                  See the app surface
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className="landing-section">
          <div className="footer-cta">
            <div>
              <p className="section-kicker">Demo Ready</p>
              <h2 className="section-title">The backend already works. Now it looks like a product.</h2>
            </div>
            <div className="inline-actions">
              <Link href="/register" className="button">
                Create account
              </Link>
              <Link href="/login" className="button-secondary">
                Sign in
              </Link>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
