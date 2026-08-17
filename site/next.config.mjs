/** @type {import('next').NextConfig} */

// Static export, so the writeup can be served from GitHub Pages with no
// runtime. BASE_PATH is set by the Pages workflow to "/<repo>", since Pages
// serves project sites from a subdirectory; local `next dev` leaves it empty.
const basePath = process.env.BASE_PATH || '';

const nextConfig = {
  output: 'export',
  basePath,
  images: { unoptimized: true },
  // Pages serves /foo as /foo/index.html; trailing slashes keep relative asset
  // paths resolving correctly under the subdirectory.
  trailingSlash: true,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
