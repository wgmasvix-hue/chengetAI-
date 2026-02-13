# CLAUDE.md

This file provides guidance for AI assistants working with the ChengetAI repository.

## Project Overview

ChengetAi is a digital agriculture intelligence platform for Zimbabwe's agricultural sector. It helps farmers, institutions (NGOs, ministries, financiers), and other stakeholders make data-driven decisions by converting scattered field data into actionable insights.

**Current status:** Prototype/experimentation phase (Phase 1). The repository currently contains a static landing page deployed via GitHub Pages.

## Repository Structure

```
chengetAI-/
├── .github/
│   └── workflows/
│       └── static.yml       # GitHub Pages deployment workflow
├── index.html                # Landing page (main application file)
├── styles.css                # Site styles (all CSS for index.html)
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
└── CLAUDE.md                 # This file
```

## Tech Stack

### Current (Phase 1)
- **Frontend:** Static HTML5 + CSS
- **Deployment:** GitHub Pages (auto-deployed from `main` branch)

### Planned (Phase 2+)
- **Backend:** FastAPI (Python) for data ingestion and analytics
- **Frontend Dashboard:** React with charts, tables, and real-time insights
- **IoT Layer:** Solar dryers, cold rooms, soil sensors, field monitoring devices
- **AI/ML:** Spoilage prediction, yield forecasting, anomaly detection

## Development Workflow

### Branching
- The default branch is `main`
- GitHub Pages deploys automatically on push to `main`

### Running Locally
No build step required. Open `index.html` directly in a browser or use any static server:
```bash
python3 -m http.server 8000
```

### Deployment
Deployment is handled by `.github/workflows/static.yml`:
1. Triggers on push to `main` or manual workflow dispatch
2. Uploads the entire repository root as a static site
3. Deploys to GitHub Pages
4. No build step required (static files served directly)

### Build Tools
There is currently no `package.json`, no bundler, no linter, and no test framework. The project is plain HTML/CSS with no build step.

## Code Conventions

### HTML
- Semantic HTML5 elements (`header`, `nav`, `section`, `footer`)
- 4-space indentation
- UTF-8 encoding
- Responsive viewport meta tag
- Section comments: `<!-- SECTION NAME -->`
- Anchor-based navigation using section `id` attributes

### CSS
- Class naming: descriptive, hyphen-separated (`header-container`, `hero-content`, `section-title`)
- Layout utilities: `container`, `grid-3`
- Component pattern: `card`, `btn`
- Modifiers: `primary`, `secondary`, `alt`
- Mobile-first responsive design with `@media` breakpoints
- Color scheme: green palette (`#1a5632` primary, `#2d8659` accent)

## Key Files

| File | Description |
|------|-------------|
| `index.html` | Landing page with platform overview, features, tech stack, and contact sections |
| `styles.css` | All styles for the landing page (layout, cards, responsive, colors) |
| `README.md` | Project vision, tech stack, roadmap, getting started guide |
| `.github/workflows/static.yml` | GitHub Pages CI/CD pipeline |
| `.gitignore` | Ignores OS files, editor files, env secrets, dependency dirs |

## Contact

Project email: info@chengetai.org
