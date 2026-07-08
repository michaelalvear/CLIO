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

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/michaelalvear/CLIO.git
cd CLIO
python -m venv venv
```

Activate the virtual environment:
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

```bash
pip install -e .
```

### 2. Get the data

CLIO needs two kinds of input data, neither of which is in this repository (they're
gitignored — see `data/` and `documents/`):

- **NetCDF dataset** — the raster hydrology data CLIO analyzes. This is our own
  model output, hosted on Zenodo.
- **RAG source documents** — PDFs embedded into a vector database that ground the
  agent's explanations in domain knowledge. These are third-party works (USGS/NOAA/
  IPCC); rather than redistribute copies, we link to their canonical DOIs.

#### NetCDF files (Zenodo)

Browse and download manually at **`<ZENODO_DOI_HERE>`**, or fetch directly:

```bash
mkdir -p data/netcdf

curl -L -o data/netcdf/baseline_surface.nc \
  "https://zenodo.org/records/<RECORD_ID>/files/baseline_surface.nc?download=1"
curl -L -o data/netcdf/farfuture585_surface.nc \
  "https://zenodo.org/records/<RECORD_ID>/files/farfuture585_surface.nc?download=1"
```

(Windows PowerShell: replace `curl -L -o <file> <url>` with
`Invoke-WebRequest <url> -OutFile <file>`.)

Pick **one** of the two `.nc` files as your active dataset for `NETCDF_DATA_PATH`
below — `baseline_surface.nc` covers 2011–2019 (historical), `farfuture585_surface.nc`
covers 2091–2099 under the SSP5-8.5 climate scenario. Both share the same grid, CRS,
and variables (`salinity`, `temperature`), so you can swap between them later without
any code changes.

#### RAG source documents (canonical sources, not mirrored)

Create `documents/rag/`, then follow each DOI to its publisher page and download the
PDF, saving it under the filename shown:

| Save as | Document | DOI |
|---|---|---|
| `bisect_paper.pdf` | Swain et al. 2019, *The Hydrologic System of the South Florida Peninsula* (USGS SIR 2019-5045) | [10.3133/sir20195045](https://doi.org/10.3133/sir20195045) |
| `IPCC_AR6_SYR_SPM.pdf` | IPCC AR6 Synthesis Report, Summary for Policymakers | [10.59327/IPCC/AR6-9789291691647.001](https://doi.org/10.59327/IPCC/AR6-9789291691647.001) |
| `noaa_18399_DS1.pdf` | Sweet et al. 2017, *Global and Regional Sea Level Rise Scenarios for the United States* (NOAA Tech. Report NOS CO-OPS 083) | [10.7289/V5/TR-NOS-COOPS-083](https://doi.org/10.7289/V5/TR-NOS-COOPS-083) |

These DOI links resolve to landing pages rather than direct PDF downloads (and the
NOAA repository in particular blocks scripted/non-browser requests), so this step is
manual rather than a `curl` one-liner.

### 3. Configure environment

Create a `.env` file in the repo root:

```
GEMINI_API_KEY=your_gemini_api_key
NETCDF_DATA_PATH=./data/netcdf/baseline_surface.nc
DOCS_PATH=./documents/rag
CHROMADB_PATH=./data/chromadb
CHROMADB_COLLECTION=domain_knowledge
```

All five variables are required at runtime.

### 4. Build the RAG vector database

```bash
python rag/rag_manager.py
```

At the prompt, run `add` to embed everything in `DOCS_PATH` into Chroma
(`CHROMADB_PATH`/`CHROMADB_COLLECTION`). This only needs to be done once — or again
if you add/change source documents. Run `list` to confirm the collection was created,
then `exit`.

### 5. Run the app

```bash
python -m clio.app
```

Open **http://localhost:5001**. Startup takes a few seconds while the dataset and
agent graph initialize.

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

data/                     Gitignored — NetCDF files (from Zenodo) + generated Chroma DB
documents/                Gitignored — RAG source PDFs (downloaded via DOI, see Quickstart)
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
licensed separately under CC-BY 4.0 — see the Zenodo record for details. The RAG
source PDFs remain under their original publishers' terms (USGS/NOAA public domain;
IPCC copyright — see each DOI landing page).

---

## For maintainers: publishing a new data version

If the underlying model output changes (e.g. a new BISECT run), upload the new
files as a **new version** of the existing Zenodo record (don't create a fresh
record — this preserves the DOI history and lets old citations still resolve).
On the record page: **New version** → replace/add files → update the version
metadata → **Publish**. Then update the `<RECORD_ID>` placeholders above.
