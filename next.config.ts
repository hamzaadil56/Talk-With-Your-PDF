import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === "development";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        // Dev: proxy to the locally running FastAPI (uvicorn on :8000).
        // Prod: route to the Python serverless function (Vercel preserves the
        // original path, so FastAPI's /api/* routes still match).
        destination: isDev
          ? "http://127.0.0.1:8000/api/:path*"
          : "/api/index",
      },
    ];
  },
};

export default nextConfig;
