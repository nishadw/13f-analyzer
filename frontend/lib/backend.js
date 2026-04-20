function getBackendBaseUrl() {
  return process.env.BACKEND_BASE_URL || "https://one3f-analyzer-6j85.onrender.com";
}

export async function fetchBackendJson(pathname) {
  const url = new URL(pathname, getBackendBaseUrl()).toString();

  const res = await fetch(url, {
    // Allow self-signed certs in local dev (Node 18+ supports this via env var)
    // In production (Render), the cert is valid so this is a no-op.
    cache: "no-store",
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Backend status ${res.status}: ${body}`);
  }

  return res.json();
}
