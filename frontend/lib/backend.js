import https from "node:https";

function getBackendBaseUrl() {
  return process.env.BACKEND_BASE_URL || "https://localhost:7779";
}

export async function fetchBackendJson(pathname) {
  const base = getBackendBaseUrl();
  const url = new URL(pathname, base);

  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        method: "GET",
        rejectUnauthorized: false,
      },
      (res) => {
        let body = "";
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          if (res.statusCode && res.statusCode >= 400) {
            reject(new Error(`Backend status ${res.statusCode}: ${body}`));
            return;
          }
          try {
            resolve(JSON.parse(body || "null"));
          } catch (err) {
            reject(err);
          }
        });
      }
    );

    req.on("error", reject);
    req.end();
  });
}
