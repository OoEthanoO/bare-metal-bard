import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'A CUDA matmul at 90% of cuBLAS',
  description:
    'Taking a hand-written CUDA SGEMM from 1.2% to 90.6% of cuBLAS on an RTX 4070 Laptop, then training a GPT on it with no PyTorch and no vendor BLAS.',
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
