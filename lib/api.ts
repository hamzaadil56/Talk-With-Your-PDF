// Typed client for the FastAPI backend. All routes are same-origin under /api,
// so no base URL is needed (dev proxies to uvicorn via next.config rewrites).

export interface IngestResponse {
  document_id: string;
  filename: string;
  chunk_count: number;
}

export interface Source {
  text: string;
  score: number;
  page: number | null;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(res: Response): Promise<never> {
  let detail = `Request failed (${res.status})`;
  try {
    const data = await res.json();
    if (data?.detail) detail = data.detail;
  } catch {
    /* non-JSON error body */
  }
  throw new ApiError(detail, res.status);
}

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/documents", { method: "POST", body: form });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function askQuestion(
  documentId: string,
  question: string,
): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, question }),
  });
  if (!res.ok) return parseError(res);
  return res.json();
}
