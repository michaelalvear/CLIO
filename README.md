# CLIO — Climate and Land data Interpretation Oracle

A research prototype exploring LLMs as natural language mediators for spatiotemporal
NetCDF climate/hydrology data, built around the BISECT South Florida hydrology model
dataset. Users draw a bounding box on a map; CLIO deterministically slices the
dataset and uses an LLM to explain the results in plain language, grounded in a
domain-knowledge RAG layer.

**Authors:** Michael Alvear, Luis Cabrera, Mark Schwartz, Christopher Smith,
Mewcha A. Gebremedhin, Ahmed S. Elshall
**Institution:** U.A. Whitaker College of Engineering & The Water School, Florida
Gulf Coast University

---

## Archived releases

| What | DOI | Contents |
|---|---|---|
| Code | [10.5281/zenodo.23742583](https://doi.org/10.5281/zenodo.23742583) | Snapshots of this repository (all versions; resolves to the latest) |
| Data | [10.5281/zenodo.23743654](https://doi.org/10.5281/zenodo.23743654) | `farfuture585_surface.nc` — BISECT surface-water salinity and temperature, 2091–2099, SSP5-8.5 (the dataset used for the paper's demonstration and evaluation) |

---

## Quickstart

Requires Python 3.10 (3.9+ should work) and a Google Gemini API key.

### 1. Get the code and install

Clone the repository:

```bash
git clone https://github.com/michaelalvear/CLIO.git
cd CLIO
```

…or download the latest archived release from
[Zenodo](https://doi.org/10.5281/zenodo.23742583) and unzip it.

Then create a virtual environment and install CLIO into it:

```bash
python -m venv venv
```

Activate it:
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

```bash
pip install -e .
```

Use this virtual environment for every command below. On Windows especially, a bare
`python` can resolve to a different system or Anaconda install that lacks CLIO or has
incompatible `numpy`/`chromadb` versions (which can crash the app with no traceback
the first time it searches the knowledge base).

### 2. Download the dataset

The NetCDF dataset is not in this repository. Download `farfuture585_surface.nc`
(~520 MB) from [Zenodo](https://doi.org/10.5281/zenodo.23743654), or fetch it directly:

```bash
mkdir -p data/netcdf
curl -L -o data/netcdf/farfuture585_surface.nc \
  "https://zenodo.org/records/23743655/files/farfuture585_surface.nc?download=1"
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data\netcdf
Invoke-WebRequest "https://zenodo.org/records/23743655/files/farfuture585_surface.nc?download=1" -OutFile data\netcdf\farfuture585_surface.nc
```

The file covers 2091–2099 under the SSP5-8.5 climate scenario and contains two
variables, `salinity` and `temperature`, on a rectilinear grid over South Florida.

### 3. Download the knowledge-base documents

The agent's explanations are grounded in four published documents. These are
third-party works, so we link to them rather than redistribute copies. Create
`documents/rag/`, follow each DOI to the publisher's page, and save the PDF directly in
that folder (not in a subfolder):

| Document | DOI |
|---|---|
| Swain, E.D., Lohmann, M.A., & Goodwin, C.R. (2019). *The Hydrologic System of the South Florida Peninsula: Development and Application of the Biscayne and Southern Everglades Coastal Transport (BISECT) Model.* USGS SIR 2019-5045 | [10.3133/sir20195045](https://doi.org/10.3133/sir20195045) |
| Gebremedhin, M.A., Elshall, A.S., Ye, M., Tsegaye, S., & Rotz, R. (2026). *Groundwater-surface water interactions and salinity responses to projected rainfall and sea-level rise in South Florida.* Environmental Research Communications | [10.1088/2515-7620/ae7c60](https://doi.org/10.1088/2515-7620/ae7c60) |
| IPCC (2023). *Climate Change 2023: Synthesis Report — Summary for Policymakers* | [10.59327/IPCC/AR6-9789291691647.001](https://doi.org/10.59327/IPCC/AR6-9789291691647.001) |
| Sweet, W.V., et al. (2017). *Global and Regional Sea Level Rise Scenarios for the United States.* NOAA Tech. Report NOS CO-OPS 083 | [10.7289/V5/TR-NOS-COOPS-083](https://doi.org/10.7289/V5/TR-NOS-COOPS-083) |

The DOIs resolve to landing pages rather than direct downloads (and the NOAA
repository blocks scripted requests), so this step is manual.

### 4. Configure the environment

Create a `.env` file in the repository root:

```
GEMINI_API_KEY=your_gemini_api_key
NETCDF_DATA_PATH=./data/netcdf/farfuture585_surface.nc
DOCS_PATH=./documents/rag
CHROMADB_PATH=./data/chromadb
CHROMADB_COLLECTION=domain_knowledge
```

All five variables are required. Relative paths are resolved from the directory you
run commands in, so run everything from the repository root (or use absolute paths).

### 5. Build the knowledge base

```bash
python rag/rag_manager.py
```

At the `rag>` prompt, run `add`. For each PDF it will ask you to confirm the title,
author, year, and DOI it extracted — these are what the agent cites, and PDF metadata
is often wrong, so replace each value with the one from the table in step 3 (press
Enter to accept a value that is already correct). Your answers are saved to
`documents/rag/metadata.json`, so later runs are pre-filled. Confirm with `y` to
embed the documents, run `list` to check that the collection exists, then `exit`.

You only need to do this once, or again after changing the documents. If you rebuild
the collection (`delete` then `add`) while the app is running, restart the app.

### 6. Run the app

```bash
python -m clio.app
```

Open **http://localhost:5001**. Startup takes a few seconds while the dataset and
agent load. Draw a rectangle on the map, choose a variable and time range, then ask
questions in the chat panel.

---

## Project Layout

```
src/clio/
  app.py                 Flask server — dataset owner, three endpoints
  cf_utils.py             CF axis/CRS detection, coordinate transforms
  data_processor.py       Deterministic Xarray slicing, heatmap, timeseries, stats
  agent/
    orchestration.py      LangGraph agent graph, system prompt, run_agent()
    tools.py               retrieve_domain_context RAG tool
    schemas.py              AgentState

frontend/
  index.html              Leaflet map + draw + dashboard + chat UI

rag/
  rag_manager.py          Interactive console for managing the Chroma vector DB

tests/
  schema_and_tools_tests.py

data/                     Gitignored — NetCDF dataset (from Zenodo) + generated Chroma DB
documents/                Gitignored — knowledge-base PDFs (downloaded via DOI, see Quickstart)
```

## Tests

```bash
pytest tests/schema_and_tools_tests.py
```

Requires `NETCDF_DATA_PATH` to be set — the `real_dataset` fixture is skipped otherwise.

## Dataset Compatibility

CLIO generalizes beyond BISECT to any **rectilinear, single-vertical-level,
CF-compliant NetCDF** dataset — CRS, X/Y/T axes, and calendars are all detected
rather than hardcoded. It does not yet support datasets with a vertical/Z axis,
curvilinear grids, or 0–360° longitude conventions.

## License

Code is licensed under AGPLv3 (see `LICENSE`). The Zenodo-hosted NetCDF dataset is
licensed separately under CC-BY 4.0 — see the [Zenodo record](https://doi.org/10.5281/zenodo.23743654).
The knowledge-base PDFs remain under their original publishers' terms (see each DOI landing page).

---

## For maintainers: publishing a new data version

If the underlying model output changes (e.g. a new BISECT run), upload the new
files as a **new version** of the existing Zenodo record (don't create a fresh
record — this preserves the DOI history and lets old citations still resolve).
On the record page: **New version** → replace/add files → update the version
metadata → **Publish**. Then update the direct-download URL in Quickstart step 2
(it points at a specific version's record, currently `23743655`). The data DOI in
this README (`10.5281/zenodo.23743654`) is the all-versions DOI and always resolves
to the latest version, so it doesn't need changing.
