# Professional report and fast-mode update

Preserves the full Python application and original knowledge collections,
diagnosis, RAG integration, storage, export, simulations and effectiveness lab.

Guided Investigation now shows a formatted HTML report with case metadata,
PFMEA investigation worksheet, recorded evidence, cause rankings, CAPA sections,
action tables and approval fields. Download the HTML and open it in a browser;
Ctrl+P saves a PDF. This layout uses deterministic escaped HTML, not another
LLM call. Report source text is available in a collapsed section. Existing Word
export remains available in Cases & Actions.

The action-plan prompt now asks for What / Where / When / Who / Required Results.
New proposed deadlines remain uncommitted in action tracking until reviewed.
Existing four-column saved plans remain compatible.

FAST MODE (default): one drafting call, temperature 0.1, 2400-token output budget,
30-second HTTP timeout, no automatic retries. Local lexical checks and citation-ID
checks remain. Multiple per-claim LLM judge calls are skipped by default; the
check panel explicitly states this is not factual verification. Set
CAPA_DEEP_VERIFY=1 in Render to opt into slower claim-level LLM checks.
No latency guarantee is possible; cold starts and provider/network load remain.
If a provider call fails, the error is shown; no template is passed off as an LLM result.

40 automated tests passed, including report escaping, professional action parsing,
fast-mode no-extra-LLM checks and repeated saved inspections. Server startup was
checked locally. Live provider generation was not tested without credentials.
The generated layout sample uses authored demonstration text, not LLM output.
Visual browser QA was unavailable because no browser binary was installed.

S/O/D ratings remain Pending: this update does not invent validated scoring.
The source registry still contains only one eligible checked steel reference;
source coverage needs improvement. None of the simulation results establishes
real-plant accuracy. This is an engineer-reviewed draft workflow.

Deploy using START_HERE_RENDER.md. For PDF download use the new formatted HTML;
the existing Word exporter retains its original layout.
