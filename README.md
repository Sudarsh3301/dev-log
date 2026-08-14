# Übermensch

A personal research notebook — terminal/notebook-styled Astro site covering AI systems, inference optimization, reasoning architectures, and multimodal AI.

## Tech Stack

- [Astro](https://astro.build/)
- [TypeScript](https://www.typescriptlang.org/)
- [TailwindCSS](https://tailwindcss.com/)
- [Pagefind](https://pagefind.app/) for static search

## Project Structure

```bash
/
├── public/
│   ├── pagefind/ # auto-generated on build
│   └── favicon.svg
├── src/
│   ├── assets/
│   │   ├── icons/
│   │   └── images/
│   ├── components/
│   ├── data/
│   │   └── blog/
│   ├── layouts/
│   ├── pages/
│   ├── scripts/
│   ├── styles/
│   ├── utils/
│   ├── config.ts
│   ├── constants.ts
│   └── content.config.ts
└── astro.config.ts
```

## Running Locally

```bash
pnpm install
pnpm run dev
```

## Commands

| Command                  | Action                                          |
| :------------------------ | :----------------------------------------------- |
| `pnpm install`            | Installs dependencies                            |
| `pnpm run dev`            | Starts local dev server at `localhost:4321`      |
| `pnpm run build`          | Build the production site to `./dist/`           |
| `pnpm run preview`        | Preview the build locally before deploying       |
| `pnpm run format:check`   | Check code format with Prettier                  |
| `pnpm run format`         | Format code with Prettier                        |
| `pnpm run sync`           | Generate TypeScript types for Astro modules      |
| `pnpm run lint`           | Lint with ESLint                                 |

## Google Site Verification (optional)

```bash
# in your environment variable file (.env)
PUBLIC_GOOGLE_SITE_VERIFICATION=your-google-site-verification-value
```
