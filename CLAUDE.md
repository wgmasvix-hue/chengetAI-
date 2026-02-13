# CLAUDE.md

This file provides guidance for AI assistants working with the ChengetAI repository.

## Project Overview

ChengetAi is a digital agriculture intelligence platform for Zimbabwe's agricultural sector. It helps farmers, institutions (NGOs, ministries, financiers), and other stakeholders make data-driven decisions by converting scattered field data into actionable insights.

**Current status:** Prototype/experimentation phase (Phase 1). The repository currently contains only a static landing page deployed via GitHub Pages.

## Repository Structure

```
chengetAI-/
├── .github/
│   └── workflows/
│       └── static.yml      # GitHub Pages deployment workflow
├── index.html               # Landing page (main application file)
├── README.md                 # Project documentation
├── Readme.hm                 # Duplicate README (identical content)
└── CLAUDE.md                 # This file
```

**Note:** `styles.css` is referenced in `index.html` but is not yet present in the repository.

## Tech Stack

### Current (Phase 1)
- **Frontend:** Static HTML5/CSS
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

### Deployment
Deployment is handled by `.github/workflows/static.yml`:
1. Triggers on push to `main` or manual workflow dispatch
2. Uploads the entire repository root as a static site
3. Deploys to GitHub Pages
4. No build step required (static files served directly)

### No Build Tools
There is currently no `package.json`, no bundler, no linter, and no test framework configured. The project is plain HTML/CSS with no build step.

## Code Conventions

### HTML
- Semantic HTML5 elements (`header`, `nav`, `section`, `footer`)
- 4-space indentation
- UTF-8 encoding
- Responsive viewport meta tag
- Section comments: `<!-- SECTION NAME -->`
- Anchor-based navigation using section `id` attributes

### CSS Class Naming
- Descriptive, hyphen-separated names: `header-container`, `hero-content`, `section-title`
- Layout utilities: `container`, `grid-3`
- Component pattern: `card`, `btn`
- Modifiers: `primary`, `secondary`, `alt`
- Section-scoped prefixes: `hero-buttons`, `contact-email`

## Key Files

| File | Description |
|------|-------------|
| `index.html` | Landing page with platform overview, features, tech stack, and contact sections |
| `README.md` | Project vision, status, roadmap, and contributing info |
| `.github/workflows/static.yml` | GitHub Pages CI/CD pipeline |

## Known Issues

- `styles.css` is linked in `index.html` but missing from the repository; the page currently renders without custom styles
- `Readme.hm` is a duplicate of `README.md` with a non-standard file extension

## Contact

Project email: info@chengetai.org
