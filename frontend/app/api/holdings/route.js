import { NextResponse } from "next/server";
import { fetchBackendJson } from "../../../lib/backend";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const cik = searchParams.get("cik") || "";
  const topN = Number(searchParams.get("top_n") || "25");

  if (!cik) {
    return NextResponse.json({ error: "Missing required query param: cik" }, { status: 400 });
  }

  try {
    const data = await fetchBackendJson(
      `/fund_holdings?cik=${encodeURIComponent(cik)}&top_n=${topN}`
    );
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to fetch holdings for ${cik}: ${String(error)}` },
      { status: 500 }
    );
  }
}
