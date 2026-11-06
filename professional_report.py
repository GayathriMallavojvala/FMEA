"""Escaped, deterministic presentation of generated content; no extra LLM call."""
import html
import re
from datetime import datetime, timezone


def esc(value):
    return html.escape(str(value), quote=True)


CSS = '''
.quality-report{font:14px/1.5 Arial,sans-serif;color:#183238;background:#fff;max-width:1200px;margin:20px auto;border:1px solid #cad5d5}
.quality-report *{box-sizing:border-box}.quality-report .report-head{background:#842f32;color:white;padding:22px 26px;display:flex;justify-content:space-between;gap:18px}
.quality-report h2{color:inherit;margin:4px 0;font-size:25px}.quality-report h3{background:#dce9e9;padding:10px 14px;margin:0;color:#183238;font-size:15px}
.quality-report p{padding:0 14px}.quality-report table{border-collapse:collapse;width:100%;table-layout:fixed;margin:0}.quality-report th,.quality-report td{border:1px solid #c3cecf;padding:9px;vertical-align:top;text-align:left;overflow-wrap:anywhere;font-size:12px}
.quality-report th{background:#f0f4f3;font-weight:bold}.quality-report .scroll{overflow-x:auto}.quality-report .section{margin:0 0 22px}.quality-report .small{font-size:12px}.quality-report .badge{font-size:12px;font-weight:bold;text-align:right}.quality-report .report-note{background:#f8f5eb;padding:12px 14px;margin:0}
@page{size:A4 landscape;margin:12mm}@media print{body{margin:0}.quality-report{border:0;max-width:none;margin:0}.quality-report tr{break-inside:avoid}.quality-report h3{break-after:avoid}.quality-report .report-head,.quality-report h3,.quality-report th{print-color-adjust:exact}.quality-report .scroll{overflow:visible}}
'''


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def prose(markdown):
    """Preserve all generated text, headings and tables while escaping HTML."""
    out, pending = [], []
    def flush():
        if pending:
            headers = pending[0]
            rows = [r for r in pending[1:] if not all(re.fullmatch(r'[:\-\s]+', c) for c in r)]
            out.append(table(headers, [(r+['']*len(headers))[:len(headers)] for r in rows]))
            pending.clear()
    for line in markdown.splitlines():
        line=line.strip()
        if line.startswith('|'):
            pending.append([c.strip() for c in line.strip('|').split('|')]);continue
        flush()
        clean=re.sub(r'^#+\s*','',line).replace('**','')
        if re.match(r'^\d+[.)]\s',clean) and len(clean)<110:
            out.append('<h3>'+esc(clean)+'</h3>')
        elif clean:
            out.append('<p>'+esc(clean)+'</p>')
    flush()
    return ''.join(out)


def render(report, state, scenario, owner='', elapsed=0):
    family=scenario['defect_type'].replace('_',' ')
    labels={c['id']:c['label'] for c in scenario['candidate_causes']}
    causes=sorted(state['belief'],key=state['belief'].get,reverse=True)
    priors=scenario.get('prior',{})
    total=sum(priors.get(c,1) for c in causes) or 1
    tests={t['id']:t for t in scenario['tests']}
    evidence=[[f'E{i}',tests[h['test_id']]['label'],h['outcome'].replace('_',' ')] for i,h in enumerate(state['history'],1)]
    return '<style>'+CSS+'</style><div class="quality-report">'+\
        '<div class="report-head"><div><small>QUALITY ASSURANCE / CONTROLLED DRAFT</small><h2>Steel quality investigation record</h2></div><div class="badge">DRAFT<br>Engineer review required</div></div>'+\
        table(['Case','Prepared UTC','Prepared by','Status'],[[state.get('case_id','Unsaved'),datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),owner or 'Not supplied','Investigation open']])+\
        '<p class="report-note">Provisional investigation record. Model ranking does not confirm a cause. Proposed actions require approval. Generation time: '+esc(f'{elapsed:.1f}')+' seconds; network and provider load vary.</p>'+\
        '<div class="section"><h3>01 · Observation and scope</h3>'+table(['Reported observation','Matched family','Scenario process stage'],[[state.get('defect_text','Not supplied'),family,scenario.get('process_stage','Not supplied').replace('_',' ')]])+'</div>'+\
        '<div class="section"><h3>02 · Process FMEA — investigation worksheet</h3>'+table(['Failure mode','Effect','Candidate cause','Prevention control','Detection control','S','O','D'],[[family,'Assess against product specification',labels[c],'Verify actual plant controls','Verify actual inspection controls','Pending','Pending','Pending'] for c in causes])+\
        '<p class="small">S = severity; O = occurrence; D = detection. Ratings and risk priority remain pending plant scoring evidence and review. Bayesian weights are separate from S/O/D.</p></div>'+\
        '<div class="section"><h3>03 · Recorded inspection evidence</h3>'+(table(['Reference','Inspection performed','Reported finding'],evidence) if evidence else '<p>No inspection finding recorded.</p>')+'</div>'+\
        '<div class="section"><h3>04 · Diagnostic ranking</h3>'+table(['Candidate cause','Starting weight','Current weight'],[[labels[c],f'{100*priors.get(c,1)/total:.1f}%',f'{100*state["belief"][c]:.1f}%'] for c in causes])+\
        '<p class="small">Weights use authored scenario assumptions. Findings are user-reported. Neither is independent confirmation of causation.</p></div>'+\
        '<div class="section"><h3>05 · CAPA investigation and action record</h3>'+prose(report)+'</div>'+\
        '<div class="section"><h3>06 · Review, effectiveness and closure</h3>'+table(['Implementation approved by / date','Effectiveness verified by / date','Closure decision'],[['Pending','Pending','Open — approval required']])+\
        '<p class="small">Use the CAPA Effectiveness Lab to assess follow-up counts separately; no effectiveness result is inferred from this draft.</p></div></div>'


def document(view):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Steel quality investigation report</title></head><body>'+view+'</body></html>'
