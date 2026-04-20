import "./globals.css";

export const metadata = {
  title: "13F Analyzer — Institutional Portfolio Intelligence",
  description: "Track 13F filings from top hedge funds, analyse position changes, and view ML-driven conviction signals in real time.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
