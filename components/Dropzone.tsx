"use client";

import { useCallback, useRef, useState } from "react";

interface DropzoneProps {
  onFile: (file: File) => void;
  busy: boolean;
}

export function Dropzone({ onFile, busy }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      onFile(files[0]);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!busy) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !busy && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !busy) inputRef.current?.click();
      }}
      className={`group flex w-full cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed px-8 py-16 text-center transition-colors ${
        dragging
          ? "border-accent bg-card"
          : "border-line bg-card/50 hover:border-accent/60"
      } ${busy ? "cursor-wait opacity-70" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={busy}
      />
      <span className="grid h-14 w-14 place-items-center rounded-full border border-line font-display text-2xl italic text-accent">
        {busy ? "…" : "¶"}
      </span>
      <p className="mt-5 font-display text-2xl tracking-tight">
        {busy ? "Indexing your document…" : "Drop a PDF to begin"}
      </p>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-ink-soft">
        {busy
          ? "Extracting text, splitting passages, and embedding them. One moment."
          : "Click to browse or drag a file here. Text-based PDFs up to 4 MB."}
      </p>
    </div>
  );
}
