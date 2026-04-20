import { NextResponse } from "next/server";
import { fetchBackendJson } from "../../../lib/backend";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const topN = Number(searchParams.get("top_n") || "50");

  try {
    const data = await fetchBackendJson(`/stock_signals?top_n=${topN}`);
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to fetch signals: ${String(error)}` },
      { status: 500 }
    );
  }
}
