"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "@/lib/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  pending?: boolean;
}

function uniquePages(sources: Source[]): number[] {
  const pages = sources
    .map((s) => s.page)
    .filter((p): p is number => typeof p === "number");
  return [...new Set(pages)].sort((a, b) => a - b);
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-ink px-4 py-3 text-paper">
          {message.content}
        </div>
      </div>
    );
  }

  const pages = message.sources ? uniquePages(message.sources) : [];

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%]">
        <span className="mb-1.5 block font-mono text-[10px] uppercase tracking-widest text-ink-soft">
          Assistant
        </span>
        <div className="rounded-2xl rounded-bl-sm border border-line bg-card px-4 py-3">
          {message.pending ? (
            <div className="flex items-center gap-1.5 py-1">
              <span className="dot" />
              <span className="dot" style={{ animationDelay: "0.2s" }} />
              <span className="dot" style={{ animationDelay: "0.4s" }} />
            </div>
          ) : (
            <div className="prose-answer text-[15px]">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
          {pages.length > 0 && (
            <span className="mt-3 block font-mono text-[10px] uppercase tracking-widest text-accent">
              cited · {pages.map((p) => `p. ${p}`).join(", ")}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
