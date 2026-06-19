import Link from "next/link";

const features = [
  {
    n: "01",
    title: "Grounded in your pages",
    body: "Every answer is retrieved from the document you uploaded — not invented. Relevant passages are pulled in before a single word is written.",
  },
  {
    n: "02",
    title: "Cited, not conjured",
    body: "Responses come back with the source passages and page numbers, so you can check the assistant against the original text.",
  },
  {
    n: "03",
    title: "Fast by design",
    body: "Server-side embeddings in Upstash Vector and Groq's LPU inference keep the whole round-trip brisk, even on long documents.",
  },
  {
    n: "04",
    title: "Nothing to install",
    body: "Drop in a PDF and start asking. No accounts, no setup — the document is indexed the moment it lands.",
  },
];

const steps = [
  { k: "Upload", v: "Hand over a PDF. We extract the text and split it into passages." },
  { k: "Index", v: "Passages are embedded and stored in a vector index, keyed to your document." },
  { k: "Interrogate", v: "Ask in plain language. We retrieve the closest passages and answer from them." },
];

export default function Landing() {
  return (
    <>
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <Link href="/" className="flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-full border border-ink/30 font-display text-lg italic">
            ¶
          </span>
          <span className="font-display text-lg tracking-tight">Talk With Your PDF</span>
        </Link>
        <Link
          href="/chat"
          className="rounded-full bg-ink px-5 py-2.5 font-mono text-xs uppercase tracking-widest text-paper transition-transform hover:-translate-y-0.5"
        >
          Open reading room →
        </Link>
      </header>

      <main className="mx-auto max-w-6xl px-6">
        {/* Hero */}
        <section className="grid gap-10 pb-20 pt-12 lg:grid-cols-12 lg:pt-20">
          <div className="lg:col-span-7">
            <p className="rise font-mono text-xs uppercase tracking-[0.3em] text-ink-soft">
              Retrieval-Augmented Reading · Folio No. 01
            </p>
            <h1
              className="rise mt-6 font-display text-5xl leading-[0.95] tracking-tight sm:text-6xl lg:text-7xl"
              style={{ animationDelay: "0.08s" }}
            >
              Ask your
              <br />
              documents
              <br />
              <span className="italic text-accent">anything.</span>
            </h1>
            <p
              className="rise mt-7 max-w-md text-lg leading-relaxed text-ink-soft"
              style={{ animationDelay: "0.16s" }}
            >
              Upload a PDF and hold a conversation with it. Every reply is drawn from the
              page in front of you — retrieved, cited, and answered in seconds.
            </p>
            <div
              className="rise mt-9 flex flex-wrap items-center gap-4"
              style={{ animationDelay: "0.24s" }}
            >
              <Link
                href="/chat"
                className="rounded-full bg-accent px-7 py-3.5 font-medium text-paper shadow-[0_10px_30px_-12px_rgba(194,65,12,0.7)] transition-transform hover:-translate-y-0.5"
              >
                Start reading →
              </Link>
              <a
                href="#how"
                className="font-mono text-xs uppercase tracking-widest text-ink-soft underline-offset-4 hover:underline"
              >
                How it works
              </a>
            </div>
          </div>

          {/* Sample exchange — a stamped folio card */}
          <div className="lg:col-span-5">
            <div
              className="rise relative rotate-1 rounded-2xl border border-line bg-card p-6 shadow-[0_30px_60px_-30px_rgba(33,27,20,0.45)]"
              style={{ animationDelay: "0.32s" }}
            >
              <div className="absolute -right-3 -top-3 grid h-16 w-16 -rotate-12 place-items-center rounded-full border-2 border-accent/60 font-mono text-[9px] uppercase leading-tight tracking-wider text-accent/80">
                indexed
              </div>
              <p className="font-mono text-[11px] uppercase tracking-widest text-ink-soft">
                annual-report.pdf · 48 pages
              </p>
              <div className="mt-5 space-y-4 text-sm">
                <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-ink px-4 py-3 text-paper">
                  What drove the change in operating margin?
                </div>
                <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-line bg-paper px-4 py-3 leading-relaxed">
                  Operating margin rose 2.4 pts to 18.1%, driven mainly by lower logistics
                  costs and a richer product mix.
                  <span className="mt-2 block font-mono text-[10px] uppercase tracking-widest text-accent">
                    cited · p. 12, p. 27
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="hairline" />

        {/* Features as a table of contents */}
        <section className="py-16">
          <div className="mb-10 flex items-baseline justify-between">
            <h2 className="font-display text-3xl italic">Contents</h2>
            <span className="font-mono text-xs uppercase tracking-widest text-ink-soft">
              Why it reads well
            </span>
          </div>
          <div className="grid gap-x-12 gap-y-10 sm:grid-cols-2">
            {features.map((f) => (
              <article key={f.n} className="group flex gap-5">
                <span className="font-mono text-sm text-accent">{f.n}</span>
                <div>
                  <h3 className="font-display text-2xl tracking-tight">{f.title}</h3>
                  <p className="mt-2 leading-relaxed text-ink-soft">{f.body}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <div className="hairline" />

        {/* How it works */}
        <section id="how" className="py-16">
          <h2 className="mb-10 font-display text-3xl italic">Three movements</h2>
          <ol className="grid gap-8 md:grid-cols-3">
            {steps.map((s, i) => (
              <li key={s.k} className="rounded-2xl border border-line bg-card p-6">
                <span className="font-mono text-xs uppercase tracking-widest text-ink-soft">
                  Step {i + 1}
                </span>
                <h3 className="mt-3 font-display text-2xl text-accent">{s.k}</h3>
                <p className="mt-2 leading-relaxed text-ink-soft">{s.v}</p>
              </li>
            ))}
          </ol>
        </section>

        {/* CTA */}
        <section className="py-16">
          <div className="relative overflow-hidden rounded-3xl bg-ink px-8 py-16 text-center text-paper">
            <h2 className="mx-auto max-w-2xl font-display text-4xl leading-tight sm:text-5xl">
              Your next answer is already
              <span className="italic text-accent"> on the page.</span>
            </h2>
            <Link
              href="/chat"
              className="mt-8 inline-block rounded-full bg-accent px-8 py-4 font-medium transition-transform hover:-translate-y-0.5"
            >
              Open the reading room →
            </Link>
          </div>
        </section>
      </main>

      <footer className="mx-auto max-w-6xl px-6 pb-10">
        <div className="hairline pt-6" />
        <div className="flex flex-col items-start justify-between gap-3 pt-6 font-mono text-[11px] uppercase tracking-widest text-ink-soft sm:flex-row sm:items-center">
          <span>Talk With Your PDF</span>
          <span>RAG · Upstash Vector · Groq · Next.js · FastAPI</span>
        </div>
      </footer>
    </>
  );
}
