import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  // Deliberately not pinned to a percentage: the number moved from 90 to 96
  // once double-buffering landed, and a title that has to be edited every time
  // a kernel improves is a title that will eventually be wrong.
  title: 'A CUDA matmul that catches cuBLAS',
  description:
    'Nine rewrites of a hand-written CUDA SGEMM, from 1.2% of cuBLAS to matching it in fp32 and passing it with tensor cores, then training a GPT on those kernels with no PyTorch and no vendor BLAS.',
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
