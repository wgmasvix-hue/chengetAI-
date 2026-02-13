# ChengetAi

> Agricultural intelligence platform for Zimbabwe — turning scattered data into clear, actionable insights.

## Table of Contents

- [Vision](#vision)
- [What ChengetAi Does](#what-chengetai-does)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Vision

To become the trusted decision layer for smallholder and commercial agriculture in Zimbabwe and beyond, by turning scattered data into clear, actionable insights.

## What ChengetAi Does

| Layer | Description |
|-------|-------------|
| **Farmer-facing tools** | Simple mobile/web interfaces that help farmers record activities, monitor fields, and receive timely alerts |
| **Institution dashboards** | Aggregated views for ministries, NGOs, and financiers to track adoption, performance, and risk |
| **Data & insight engine** | Ingestion of field data, weather, and market signals to power scoring, alerts, and reporting |

## Tech Stack

| Component | Technology | Status |
|-----------|-----------|--------|
| Landing page | HTML5, CSS | Active |
| Deployment | GitHub Pages | Active |
| Backend API | FastAPI (Python) | Planned |
| Dashboard | React | Planned |
| IoT devices | Solar dryers, cold rooms, soil sensors | Planned |
| AI/ML | Spoilage prediction, yield forecasting | Planned |

## Project Structure

```
chengetAI-/
├── .github/
│   └── workflows/
│       └── static.yml       # GitHub Pages deployment
├── index.html                # Landing page
├── styles.css                # Site styles
├── CLAUDE.md                 # AI assistant guide
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Getting Started

This is a static site — no build step required.

**View locally:**

1. Clone the repository:
   ```bash
   git clone https://github.com/wgmasvix-hue/chengetAI-.git
   cd chengetAI-
   ```

2. Open `index.html` in your browser, or serve with any static server:
   ```bash
   python3 -m http.server 8000
   ```

3. Visit `http://localhost:8000`

**Deployment:** Pushes to `main` automatically deploy to GitHub Pages via the workflow in `.github/workflows/static.yml`.

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Concept landing page, stakeholder demos, data model sketch | In progress |
| **Phase 2** | Farmer and institution dashboards (MVP) | Planned |
| **Phase 3** | Scoring, alerts, and integrations with partner systems | Planned |

## Contributing

This is currently an internal project. If you're interested in collaborating, please reach out to the ChengetAi team.

## License

TBD -- license will be defined once the platform structure is finalized.

## Contact

For partnerships, demos, or institutional deployments: **info@chengetai.org**
