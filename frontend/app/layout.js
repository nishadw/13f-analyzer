import "./globals.css";

export const metadata = {
  title: "13F Analyzer Dashboard",
  description: "Fund summary and per-fund current holdings tables",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
