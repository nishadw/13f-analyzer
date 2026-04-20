function getBackendBaseUrl() {
  if (process.env.BACKEND_BASE_URL) return process.env.BACKEND_BASE_URL;
  if (process.env.NODE_ENV !== "production") return "https://localhost:7779";
  return "https://one3f-analyzer-6j85.onrender.com";
}

// Render free tier spins down after inactivity; retry on 502/503 with backoff.
export async function fetchBackendJson(pathname, { retries = 3, baseDelayMs = 4000 } = {}) {
  const url = new URL(pathname, getBackendBaseUrl()).toString();

  for (let attempt = 0; attempt <= retries; attempt++) {
    const res = await fetch(url, { cache: "no-store" });

    if (res.ok) return res.json();

    const isRetryable = res.status === 502 || res.status === 503;
    if (!isRetryable || attempt === retries) {
      const body = await res.text().catch(() => "");
      const compactBody = body.replace(/\s+/g, " ").slice(0, 240);
      throw new Error(`Backend status ${res.status}: ${compactBody}`);
    }

    // Exponential backoff: 4s, 8s, 16s
    await new Promise((r) => setTimeout(r, baseDelayMs * 2 ** attempt));
  }
}
