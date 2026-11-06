# Original Steel Quality Interface + RAG / LLM

## What changed
The default interface now reuses the earlier index.html and its exact inline CSS.
The input remains Observed defect, Generate draft, and Try scratched-coil example.
Report order stays PFMEA -> CAPA -> Investigation evidence -> Inspection controls.
The red report headers, blue-grey bands, tables, metadata fields and print layout
remain. The offline caption is corrected because generation now uses a server.
Reference details are collapsed within the existing evidence annex.

Behind this interface: source-aware TF-IDF retrieval, structured LLM drafting,
typo-normalized intake, character TF-IDF candidate retrieval, and constrained LLM
interpretation of unfamiliar descriptions. Original observations are preserved.
Unclear crack patterns ask a question; unsupported defects are not forced into a
known family. No finite system can guarantee every arbitrary input is understood.

Bayesian updates are deterministic using the original authored parameters.
Inspection findings are saved first. Reports regenerate using the new revision.
If generation fails, saved evidence remains; the old report is marked outdated
and its print button disabled until regenerated. Additional inspections remain
available after model confidence thresholds. The reports are drafts for review.

## Deploy on the existing Render Python Web Service
1. Upload extracted contents to the repository, retaining folders; overwrite old
   matching files. Do not delete the Render service or put the ZIP itself in GitHub.
2. Root Directory: blank. Build: pip install -r requirements.txt
3. Start Command: python desk_server.py
4. Environment: GROQ_API_KEY = your private key; LLM_MODEL = openai/gpt-oss-20b
   Set or replace the old LLM_MODEL value in Render. Keep secrets out of GitHub.
5. Deploy latest commit, then hard-refresh the website (Ctrl+Shift+R).

Do not run python app.py for this interface: app.py is the preserved research UI.
The included render.yaml uses desk_server.py, but existing manual Render service
settings must be changed in Render, not merely by uploading render.yaml.
Optional: LLM_BASE_URL / LLM_API_KEY for another compatible provider.

## Storage
Use a Render persistent disk mounted at /var/data and CAPA_DB=/var/data/capa.db
for cases to survive restarts/deployments. Otherwise SQLite runtime changes are
ephemeral. Existing research data/capa.db is retained from the supplied archive;
use demonstration information only. New desk cases use random access tokens
stored in the browser session. This is not a user-authenticated enterprise system.

## Verification
49 Python tests passed. A served-page jsdom test checked original CSS equality,
original entry controls, three report titles and order, FMEA columns, preserved
raw input, typo intake, two sequential saved inspections, regenerated reports and
printing availability. Generation was mocked; no live provider output or latency
was verified without credentials. No full visual browser comparison was performed.
The CSS is byte-for-byte identical to the original frontend.

Run: python -m unittest discover -s tests -v

## Evidence scope
RAG passes eligible retrieved reference passages, authored scenario background and
recorded inspection findings separately to the LLM. Only one checked steel passage
currently qualifies in the inherited registry. Missing reference coverage is shown
rather than silently promoting unverified NASA/FDA/steel paraphrases to confirmed
facts. All original knowledge files, simulations, effectiveness lab, research UI,
Word-export code, cases/actions engine and benchmarks are retained in the package.

The report schema checks field types, cause-ID coverage and allowed reference IDs.
These checks do not establish factual correctness. S/O/D remain dashes pending
plant-specific scoring evidence, matching the earlier report. Diagnosis likelihoods
remain authored estimates, not measurements learned from plant incidents.
