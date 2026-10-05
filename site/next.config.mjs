/** @type {import('next').NextConfig} */

// A static export, served straight off disk by Caddy on the home server
// (deploy/windows/). The page is one route, prerendered at build time, with no
// API routes, no image optimisation and nothing read per request -- so there
// is no reason to keep a Node process running to serve it.
//
// It used to be the opposite. On Vercel no `output` was set, because Vercel
// builds and serves Next.js natively; before that, a GitHub Pages deploy needed
// `output: 'export'` plus a basePath, because Pages serves project sites from
// /<repo>. Caddy serves this one from the domain root, so it takes the export
// and not the basePath.
const nextConfig = {
  output: 'export',
};

export default nextConfig;
