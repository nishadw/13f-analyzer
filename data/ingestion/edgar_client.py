"""
SEC EDGAR REST API client.

Uses the official EDGAR submissions API as the single source of truth.
The submissions JSON includes `primaryDocument` for each filing, which is
the exact XML filename — no directory scraping or filename guessing required.

URL patterns:
  submissions:  https://data.sec.gov/submissions/CIK{10-digit-padded}.json
  infotable:    https://www.sec.gov/Archives/edgar/data/{CIK_no_zeros}/{acc_no_dashes}/{primaryDocument}
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

import config

_HEADERS = {"User-Agent": config.EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_RATE_LIMIT_DELAY = 0.12  # EDGAR fair-use: ≤ 10 req/s


def _get(url: str) -> requests.Response:
    time.sleep(_RATE_LIMIT_DELAY)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def _cik_numeric(cik: str) -> str:
    """Return the numeric CIK string with no leading zeros (used in archive paths)."""
    s = str(cik).strip()
    return str(int(s)) if s.lstrip("0") else "0"


def _cik_padded(cik: str) -> str:
    """Return the 10-digit zero-padded CIK (used in submissions API URL)."""
    return str(cik).strip().zfill(10)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_submissions(cik: str) -> dict:
    """Fetch the EDGAR submissions JSON for a CIK."""
    url = f"{config.EDGAR_SUBMISSIONS_URL}/CIK{_cik_padded(cik)}.json"
    return _get(url).json()


def fetch_fund_name(cik: str) -> str:
    """Return the fund's legal name from EDGAR submissions JSON."""
    try:
        data = _get_submissions(cik)
        return data.get("name", cik)
    except Exception:
        return cik


def _extract_13f_filings(section: dict) -> list[dict]:
    """
    Extract 13F-HR filing rows from a submissions JSON 'recent' block.
    Captures primaryDocument so we know the exact infotable XML filename.
    """
    form_types    = section.get("form", [])
    accessions    = section.get("accessionNumber", [])
    filing_dates  = section.get("filingDate", [])
    report_dates  = section.get("reportDate", [])
    primary_docs  = section.get("primaryDocument", [])

    # primaryDocument may be absent in older paginated files — pad with ""
    n = len(form_types)
    if len(primary_docs) < n:
        primary_docs = list(primary_docs) + [""] * (n - len(primary_docs))

    out = []
    for ftype, acc, fd, rd, pdoc in zip(
        form_types, accessions, filing_dates, report_dates, primary_docs
    ):
        if ftype in ("13F-HR", "13F-HR/A"):
            out.append({
                "accessionNumber": acc,
                "filingDate":      fd,
                "reportDate":      rd,
                "formType":        ftype,
                "primaryDocument": pdoc,
            })
    return out


def list_13f_filings(cik: str, n_quarters: int = 8) -> list[dict]:
    """
    Return the most recent `n_quarters` 13F-HR filings for a CIK, sorted newest-first.
    Each dict: accessionNumber, filingDate, reportDate, formType, primaryDocument.
    """
    data = _get_submissions(cik)
    filings = _extract_13f_filings(data.get("filings", {}).get("recent", {}))

    # Paginated older filings live in separate JSON files under filings.files.
    # For the most recent n_quarters we likely won't need them, but fetch if short.
    if len(filings) < n_quarters:
        for extra in data.get("filings", {}).get("files", []):
            fname = extra.get("name", "")
            if not fname:
                continue
            try:
                extra_data = _get(f"{config.EDGAR_SUBMISSIONS_URL}/{fname}").json()
                filings.extend(_extract_13f_filings(extra_data))
            except Exception:
                continue
            if len(filings) >= n_quarters:
                break

    filings.sort(key=lambda x: x["reportDate"], reverse=True)
    return filings[:n_quarters]


def _fetch_infotable_xml(cik: str, accession: str, primary_document: str = "") -> Optional[str]:
    """
    Fetch the 13F infotable XML for a filing.

    Strategy:
    1. Try primaryDocument from the submissions JSON.
       Some filers use a path like "xslForm13F_X02/primary_doc.xml" — strip
       any leading directory component and try the bare filename too.
    2. Try a short list of common canonical filenames.
    3. Scrape the HTML directory listing and try every .xml, checking for
       <infoTable> content — catches custom names like SALP_13FQ425.xml or
       renaissance13Fq42025_holding.xml.
    Archive path always uses the numeric CIK (no leading zeros).
    """
    import re as _re

    cik_num   = _cik_numeric(cik)
    acc_clean = accession.replace("-", "")
    base      = f"{config.EDGAR_ARCHIVES_URL}/{cik_num}/{acc_clean}"

    import re as _re

    def _try(filename: str) -> Optional[str]:
        if not filename:
            return None
        try:
            resp = _get(f"{base}/{filename}")
            if resp.status_code == 200 and resp.text.strip():
                text = resp.text
                # Match <infoTable>, <ns1:infoTable>, or any prefixed variant
                if _re.search(r"<\w*:?infoTable", text, _re.IGNORECASE):
                    return text
        except Exception:
            pass
        return None

    # 1. primaryDocument (and its bare filename if it contains a path separator)
    if primary_document:
        result = _try(primary_document)
        if result:
            return result
        bare = primary_document.split("/")[-1]
        if bare != primary_document:
            result = _try(bare)
            if result:
                return result

    # 2. Common canonical filenames
    for name in ("infotable.xml", "form13fInfoTable.xml", "13F_infotable.xml",
                 "informationtable.xml"):
        result = _try(name)
        if result:
            return result

    # 3. Scan HTML directory listing — catches any custom filename.
    #    Skip XBRL viewer/taxonomy artifacts (R\d+.xml, FilingSummary, etc.)
    _SKIP = _re.compile(
        r"(^R\d+\.xml$|FilingSummary|MetaLinks|\.xsd$|"
        r"\.cal\.xml|\.lab\.xml|\.pre\.xml|\.def\.xml|-index\.xml|primary_doc\.xml)",
        _re.IGNORECASE,
    )
    try:
        listing = _get(f"{base}/").text
        # Extract bare filenames from hrefs
        filenames = _re.findall(r'href="[^"]*?/([^"/]+\.xml)"', listing)
        # Try files with holdings-like keywords first
        _PRIORITY = ("infotable", "info_table", "holding", "13f")
        filenames.sort(key=lambda n: (0 if any(p in n.lower() for p in _PRIORITY) else 1, n))
        for fname in filenames:
            if _SKIP.search(fname):
                continue
            result = _try(fname)
            if result:
                return result
    except Exception:
        pass

    return None


def _strip_namespaces(xml_text: str) -> str:
    """Remove all XML namespace declarations and prefixes so XPath works plainly."""
    import re as _re
    # Drop all xmlns:prefix="..." and xmlns="..." declarations.
    # Keep namespaced tag names intact until after this pass so we can strip
    # both element and attribute prefixes in a second pass.
    xml_text = _re.sub(r'\s+xmlns(?::[A-Za-z_][\w.-]*)?="[^"]*"', "", xml_text)
    # Remove namespace prefixes from element/attribute names: <ns1:foo> → <foo>
    xml_text = _re.sub(r"(</?)\w+:", r"\1", xml_text)
    xml_text = _re.sub(r"\s\w+:(\w+)=", r" \1=", xml_text)
    return xml_text


def _parse_infotable(xml_text: str, cik: str, period: str) -> pd.DataFrame:
    """Parse an EDGAR 13F infotable XML into a tidy DataFrame."""
    xml_text = _strip_namespaces(xml_text)
    root = ET.fromstring(xml_text)

    rows = []
    for entry in root.findall(".//infoTable"):
        name  = entry.findtext("nameOfIssuer", "").strip()
        cusip = entry.findtext("cusip", "").strip()
        value = entry.findtext("value")
        shares_el = entry.find("shrsOrPrnAmt")
        shares    = shares_el.findtext("sshPrnamt") if shares_el is not None else None
        rows.append({
            "cik":                   cik,
            "period":                period,
            "name":                  name,
            "cusip":                 cusip,
            "value_thousands":       int(value) if value else None,
            "shares":                int(shares) if shares else None,
            "investment_discretion": entry.findtext("investmentDiscretion", "").strip(),
            "put_call":              entry.findtext("putCall", "").strip(),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["value_thousands"] = pd.to_numeric(df["value_thousands"], errors="coerce")
    total = df["value_thousands"].sum()
    df["weight"] = df["value_thousands"] / total if total > 0 else 0.0
    return df


def fetch_13f_history(cik: str, n_quarters: int = 8) -> pd.DataFrame:
    """
    Fetch the last `n_quarters` 13F filings for a CIK.
    Returns a single DataFrame of all holdings across all periods.
    """
    filings = list_13f_filings(cik, n_quarters=n_quarters)
    all_frames: list[pd.DataFrame] = []

    for filing in filings:
        acc    = filing["accessionNumber"]
        period = filing["reportDate"]
        pdoc   = filing.get("primaryDocument", "")

        xml_text = _fetch_infotable_xml(cik, acc, primary_document=pdoc)
        if xml_text is None:
            print(f"[edgar] WARNING: no infotable found for CIK={cik} acc={acc} primary={pdoc!r}")
            continue

        df = _parse_infotable(xml_text, cik, period)
        if not df.empty:
            df["filing_date"] = filing["filingDate"]
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result["period"]       = pd.to_datetime(result["period"])
    result["filing_date"]  = pd.to_datetime(result["filing_date"])
    return result.sort_values(["period", "cik"], ascending=[False, True])


def fetch_8k_filings(cik: str, n: int = 20) -> list[dict]:
    """Return recent 8-K filing metadata for a CIK."""
    data   = _get_submissions(cik)
    recent = data.get("filings", {}).get("recent", {})

    filings = []
    for ftype, acc, fd in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
    ):
        if ftype == "8-K":
            filings.append({"accessionNumber": acc, "filingDate": fd})
        if len(filings) >= n:
            break
    return filings
