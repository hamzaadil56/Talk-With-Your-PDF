"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Dropzone } from "@/components/Dropzone";
import { MessageBubble, type ChatMessage } from "@/components/MessageBubble";
import { askQuestion, uploadDocument, type IngestResponse } from "@/lib/api";

const uid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

export default function ChatPage() {
  const [doc, setDoc] = useState<IngestResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleFile(file: File) {
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      setDoc(result);
      setMessages([
        {
          id: uid(),
          role: "assistant",
          content: `**${result.filename}** is indexed — ${result.chunk_count} passages ready. Ask me anything about it.`,
        },
      ]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || !doc || sending) return;

    const pendingId = uid();
    setMessages((m) => [
      ...m,
      { id: uid(), role: "user", content: question },
      { id: pendingId, role: "assistant", content: "", pending: true },
    ]);
    setInput("");
    setSending(true);

    try {
      const res = await askQuestion(doc.document_id, question);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId
            ? { ...msg, content: res.answer, sources: res.sources, pending: false }
            : msg,
        ),
      );
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Something went wrong.";
      setMessages((m) => m.filter((msg) => msg.id !== pendingId));
      toast.error(detail);
    } finally {
      setSending(false);
    }
  }

  function reset() {
    setDoc(null);
    setMessages([]);
    setInput("");
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="mx-auto flex w-full max-w-3xl items-center justify-between px-6 py-5">
        <Link href="/" className="flex items-center gap-3">
          <span className="grid h-8 w-8 place-items-center rounded-full border border-ink/30 font-display italic">
            ¶
          </span>
          <span className="font-display text-base tracking-tight">Talk With Your PDF</span>
        </Link>
        {doc && (
          <button
            onClick={reset}
            className="rounded-full border border-line px-4 py-2 font-mono text-[11px] uppercase tracking-widest text-ink-soft transition-colors hover:border-accent hover:text-accent"
          >
            New document
          </button>
        )}
      </header>

      <div className="hairline mx-auto w-full max-w-3xl" />

      {!doc ? (
        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-6 py-10">
          <p className="mb-6 font-mono text-xs uppercase tracking-[0.3em] text-ink-soft">
            The reading room
          </p>
          <Dropzone onFile={handleFile} busy={uploading} />
        </main>
      ) : (
        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col overflow-hidden px-6">
          <div className="flex items-center gap-2 py-4 font-mono text-[11px] uppercase tracking-widest text-ink-soft">
            <span className="dot" style={{ animation: "none", opacity: 1 }} />
            {doc.filename} · {doc.chunk_count} passages
          </div>

          <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto pb-6">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
          </div>

          <form
            onSubmit={handleSend}
            className="sticky bottom-0 flex items-end gap-3 bg-paper/90 py-4 backdrop-blur"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend(e as unknown as React.FormEvent);
                }
              }}
              rows={1}
              placeholder="Ask this document a question…"
              className="max-h-40 flex-1 resize-none rounded-2xl border border-line bg-card px-4 py-3 leading-relaxed outline-none transition-colors focus:border-accent"
            />
            <button
              type="submit"
              disabled={sending || !input.trim()}
              className="rounded-2xl bg-accent px-5 py-3 font-medium text-paper transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
            >
              {sending ? "…" : "Ask"}
            </button>
          </form>
        </main>
      )}
    </div>
  );
}
