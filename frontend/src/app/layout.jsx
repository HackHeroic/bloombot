import './globals.css';

export const metadata = {
  title: 'BloomBot',
  description: 'Classify exam questions by Bloom level and rebalance a paper, with every rewrite checked.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" />
      </head>
      <body>{children}</body>
    </html>
  );
}
