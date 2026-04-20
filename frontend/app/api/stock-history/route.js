import { NextResponse } from "next/server";
import { fetchBackendJson } from "../../../lib/backend";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const cik = searchParams.get("cik") || "";
  const cusip = searchParams.get("cusip") || "";
  const name = searchParams.get("name") || "";
  const ticker = searchParams.get("ticker") || "";

  if (!cik) {
    return NextResponse.json({ error: "Missing required query param: cik" }, { status: 400 });
  }

  try {
    const data = await fetchBackendJson(
      `/stock_holdings_history?cik=${encodeURIComponent(cik)}&cusip=${encodeURIComponent(cusip)}&name=${encodeURIComponent(name)}&ticker=${encodeURIComponent(ticker)}&n_periods=4`
    );
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to fetch stock history for ${cik}: ${String(error)}` },
      { status: 500 }
    );
  }
}