import { NextResponse } from "next/server";
import { fetchBackendJson } from "../../../lib/backend";

export async function GET() {
  try {
    const data = await fetchBackendJson("/funds_summary");
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to fetch funds summary: ${String(error)}` },
      { status: 500 }
    );
  }
}
