import { NextResponse } from "next/server";
import fs from "node:fs/promises";
import path from "node:path";
import { fetchBackendJson } from "../../../lib/backend";

async function loadLocalSignals(topN) {
  const candidates = [
    path.resolve(process.cwd(), "data", "signals_cache.json"),
    path.resolve(process.cwd(), "..", "data", "signals_cache.json"),
  ];

  for (const filePath of candidates) {
    try {
      const raw = await fs.readFile(filePath, "utf8");
      const parsed = JSON.parse(raw);
      return {
        data: Array.isArray(parsed?.data) ? parsed.data.slice(0, topN) : [],
        columnsDefs: Array.isArray(parsed?.columnsDefs) ? parsed.columnsDefs : [],
        localFallback: true,
      };
    } catch {
      // Try next candidate path.
    }
  }

  return null;
}

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const rawTopN = Number(searchParams.get("top_n") || "20");
  const topN = Math.max(10, Math.min(20, Number.isFinite(rawTopN) ? rawTopN : 20));

  try {
    const data = await fetchBackendJson(`/stock_signals?top_n=${topN}`);
    return NextResponse.json({
      data: Array.isArray(data?.data) ? data.data.slice(0, topN) : [],
      columnsDefs: Array.isArray(data?.columnsDefs) ? data.columnsDefs : [],
    });
  } catch (error) {
    const fallback = await loadLocalSignals(topN);
    if (fallback) {
      return NextResponse.json(fallback);
    }

    return NextResponse.json(
      { error: `Failed to fetch signals: ${String(error)}` },
      { status: 500 }
    );
  }
}
