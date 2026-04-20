import { NextResponse } from "next/server";
import { fetchBackendJson } from "../../../lib/backend";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const cik = searchParams.get("cik") || "";
  const topN = Number(searchParams.get("top_n") || "30");
  const includeCandidates = searchParams.get("include_candidates") !== "false";

  if (!cik) {
    return NextResponse.json({ error: "Missing required query param: cik" }, { status: 400 });
  }

  try {
    const data = await fetchBackendJson(
      `/stock_signals?cik=${encodeURIComponent(cik)}&top_n=${topN}&include_candidates=${includeCandidates}`
    );
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to fetch signals for ${cik}: ${String(error)}` },
      { status: 500 }
    );
  }
}
