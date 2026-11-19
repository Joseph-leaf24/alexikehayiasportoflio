# syntax=docker/dockerfile:1
# The portfolio as a self-contained container, for running it locally (separate from
# the GitHub Pages deploy):
#
#   docker compose up --build        then open http://localhost:8080/alexikehayiasportoflio/
#
# or without Compose:
#
#   docker build -t alexi-portfolio .
#   docker run --rm -p 8080:8080 alexi-portfolio
#
# Base images are pinned by digest; Dependabot proposes updates (.github/dependabot.yml).

# ---- 1. Build: exact dependencies from the lockfile, then the static site ----------
FROM node:22.23.5-alpine@sha256:1314fb30ec37592d5845e5a160d522c0e680972aa534549940b8d73bde88481a AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html vite.config.ts tailwind.config.ts postcss.config.js tsconfig.json tsconfig.app.json tsconfig.node.json ./
COPY public ./public
COPY src ./src
RUN npm run build

# ---- 2. Serve: unprivileged nginx (runs as a non-root user on port 8080) ----------
FROM nginxinc/nginx-unprivileged:1.30-alpine@sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
# Same URL path as on GitHub Pages, because the build uses that base path
COPY --from=build /app/dist /usr/share/nginx/html/alexikehayiasportoflio
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD ["wget", "-q", "--spider", "http://127.0.0.1:8080/alexikehayiasportoflio/"]
