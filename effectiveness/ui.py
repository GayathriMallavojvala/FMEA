"""Gradio tab: CAPA Effectiveness Lab (design, verify, benchmark)."""
import numpy as np
import pandas as pd
import gradio as gr
from . import engine, benchmark

CSS = """
:root{--navy:#10263B;--teal:#0B8574;--mint:#5FD4BE;--amber:#B7791F;--red:#C0392B;--green:#1E8E5A}
.gradio-container{max-width:1180px!important;font-family:'Inter','DM Sans',system-ui,sans-serif}
.hero{background:linear-gradient(120deg,#10263B 0%,#1B3650 60%,#0B8574 140%);color:#F5F3EE;border-radius:18px;padding:30px 34px;margin-bottom:14px}
.hero .k{letter-spacing:.14em;font-size:12px;font-weight:700;color:#5FD4BE}
.hero h1{margin:6px 0 8px;font-size:34px;color:#fff}.hero p{margin:0;color:#C9D6E2;font-size:15px;max-width:820px}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.chip{background:#ffffff1a;border:1px solid #ffffff33;border-radius:999px;padding:5px 14px;font-size:13px;color:#fff}
.card{border-radius:16px;padding:22px 26px;border:1px solid #E3E0D8;background:#fff;box-shadow:0 2px 10px #10263B10}
.card .lab{font-size:12px;font-weight:700;letter-spacing:.1em;color:#5A6573}.card .big{font-size:44px;font-weight:800;line-height:1.1;margin:4px 0}
.card .sub{color:#33414f;font-size:15px}.bar{height:14px;border-radius:9px;background:#E9EDF1;margin:14px 0 6px;overflow:hidden}.bar>div{height:100%;border-radius:9px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px}.kpi{background:#F5F3EE;border-radius:12px;padding:12px 14px}.kpi b{display:block;font-size:20px;color:#10263B}.kpi span{font-size:12px;color:#5A6573}
.CLOSE{border-left:8px solid #1E8E5A}.REOPEN{border-left:8px solid #C0392B}.EXTEND,.ESCALATE{border-left:8px solid #B7791F}.MONITOR{border-left:8px solid #8795A5}
"""
HERO = """<div class="hero"><div class="k">STEEL QUALITY ENGINEERING · CAPA EFFECTIVENESS LAB</div>
<h1>Did the corrective action really work?</h1>
<p>Most plants close a CAPA after a fixed 30/60/90-day look at the defect count. This lab designs the monitoring window
statistically and gives a Bayesian, evidence-based decision: close, keep monitoring, or reopen.</p>
<div class="chips"><span class="chip">Gamma-Poisson model</span><span class="chip">Overdispersion-aware</span><span class="chip">Weekly sequential look</span><span class="chip">Simulation benchmark</span></div></div>"""
COL = {"CLOSE": "#1E8E5A", "REOPEN": "#C0392B", "EXTEND": "#B7791F", "ESCALATE": "#B7791F", "MONITOR": "#8795A5"}


def _nums(t):
    return [float(x) for x in str(t).replace("\n", ",").replace(";", ",").split(",") if x.strip()]


def do_design(rate, target, expected, conf, cv, pre_weeks):
    best, rows = engine.design(rate, target / 100, expected / 100, conf / 100, cv, int(pre_weeks), reps=1200)
    alt, _ = engine.design(rate, target / 100, expected / 100, conf / 100, cv, 26, reps=1200)
    df = pd.DataFrame(rows)
    long = pd.concat([df.assign(m="Chance of closing a truly effective CAPA", v=df.power * 100),
                      df.assign(m="Chance of wrongly closing an ineffective CAPA", v=df.false_close * 100)])[["weeks", "m", "v"]]
    wk = f"{best} weeks" if best else "> 26 weeks"
    tip = (f"With a 26-week baseline the same check needs only <b>{alt} weeks</b>: a longer, cleaner baseline is the cheapest way to shorten monitoring."
           if alt and (not best or alt < best) else "A longer baseline would not shorten this plan.")
    html = f"""<div class="card EXTEND" style="border-left-color:#0B8574"><div class="lab">RECOMMENDED MONITORING WINDOW</div>
<div class="big" style="color:#0B8574">{wk}</div>
<div class="sub">Closes a CAPA that truly cuts defects by {expected:.0f}% at least 80% of the time, and wrongly closes a useless one at most 5% of the time.<br>{tip}</div>
<div class="kpis"><div class="kpi"><b>{rate:g}/wk</b><span>baseline defects</span></div><div class="kpi"><b>{target:.0f}%</b><span>target reduction</span></div>
<div class="kpi"><b>{conf:.0f}%</b><span>required confidence</span></div><div class="kpi"><b>{int(pre_weeks)} wk</b><span>baseline length</span></div></div></div>"""
    plan = (f"**Effectiveness check plan (auto-drafted)**\n\n- Metric: weekly defect count for the CAPA's defect code\n- Baseline: last {int(pre_weeks)} weeks before implementation\n"
            f"- Pass rule: at least {conf:.0f}% probability that the weekly rate fell by {target:.0f}% or more\n- Window: review weekly from week 4, up to {wk}\n"
            "- Fail rule: reopen if the probability of meeting the target falls to 5% or less\n- Sign-off: quality engineer approves closure; the model only advises")
    return html, long, plan


def do_verify(pre, post, target, conf):
    pre, post = _nums(pre), _nums(post)
    if len(pre) < 3 or len(post) < 1:
        raise gr.Error("Enter at least 3 baseline weeks and 1 week after the action.")
    r = engine.verify(pre, post, target / 100, conf / 100)
    c = COL[r["decision"]]
    html = f"""<div class="card {r['decision']}"><div class="lab">DECISION</div><div class="big" style="color:{c}">{r['decision']}</div>
<div class="sub">{r['reason']}</div><div class="bar"><div style="width:{r['prob']*100:.0f}%;background:{c}"></div></div>
<div class="sub" style="font-size:13px">Probability the {target:.0f}% target is met: <b>{r['prob']:.0%}</b> (close at {conf:.0f}%)</div>
<div class="kpis"><div class="kpi"><b>{r['pre_rate']:.1f} → {r['post_rate']:.1f}</b><span>defects / week</span></div>
<div class="kpi"><b>{r['reduction_mean']:.0%}</b><span>estimated reduction</span></div>
<div class="kpi"><b>{r['ci_low']:.0%} to {r['ci_high']:.0%}</b><span>90% credible range</span></div>
<div class="kpi"><b>{r['phi']:.2f}</b><span>variability factor</span></div></div></div>"""
    d = pd.DataFrame({"week": list(range(1, len(pre) + 1)) + list(range(len(pre) + 1, len(pre) + len(post) + 1)),
                      "defects": pre + post, "phase": ["Before CAPA"] * len(pre) + ["After CAPA"] * len(post)})
    return html, d


def do_bench(rate, target, conf, cv, reps, c_fc, c_miss, c_week):
    df, s = benchmark.run(rate, target / 100, conf / 100, cv, reps=int(reps))
    s["expected_cost"] = s.false_closure * c_fc + s.missed_effect * c_miss + s.mean_weeks * c_week
    best = s.loc[s.expected_cost.idxmin(), "method"]
    out = s.rename(columns={"method": "Method", "false_closure": "False closure", "missed_effect": "Missed effect",
                            "mean_weeks": "Weeks to decide", "expected_cost": "Expected cost / CAPA (Rs)"})
    for k in ("False closure", "Missed effect"):
        out[k] = (out[k] * 100).round(1).astype(str) + "%"
    out["Weeks to decide"] = out["Weeks to decide"].round(1)
    out["Expected cost / CAPA (Rs)"] = out["Expected cost / CAPA (Rs)"].round(0).astype(int)
    bar = pd.concat([s.assign(metric="False closure %", value=s.false_closure * 100),
                     s.assign(metric="Missed effect %", value=s.missed_effect * 100)])[["method", "metric", "value"]]
    det = df.assign(closed=(df.closed * 100).round(1), weeks=df.weeks.round(1))[["method", "scenario", "closed", "weeks"]]
    det.columns = ["Method", "Scenario", "Closed as effective (%)", "Weeks"]
    return f"**Lowest expected cost under your cost settings:** {best}", out, bar, det


def build_tab():
    gr.HTML(HERO)
    with gr.Tabs():
        with gr.Tab("1 · Design the check"):
            with gr.Row():
                with gr.Column(scale=2):
                    rate = gr.Slider(1, 30, 6, step=0.5, label="Baseline defects per week")
                    tgt = gr.Slider(20, 90, 50, step=5, label="Target reduction promised by the CAPA (%)")
                    exp = gr.Slider(20, 95, 70, step=5, label="Reduction you realistically expect (%)")
                    conf = gr.Slider(80, 99, 90, step=1, label="Confidence needed to close (%)")
                    cv = gr.Slider(0.05, 0.8, 0.35, step=0.05, label="Week-to-week variability (CV)")
                    pw = gr.Slider(6, 26, 12, step=1, label="Baseline weeks available")
                    b1 = gr.Button("Design monitoring plan", variant="primary")
                with gr.Column(scale=3):
                    o1 = gr.HTML()
                    p1 = gr.LinePlot(x="weeks", y="v", color="m", y_title="Probability (%)", x_title="Weeks monitored", height=300)
                    t1 = gr.Markdown()
            b1.click(do_design, [rate, tgt, exp, conf, cv, pw], [o1, p1, t1])
        with gr.Tab("2 · Verify a CAPA"):
            with gr.Row():
                with gr.Column(scale=2):
                    pre = gr.Textbox("6, 7, 5, 8, 6, 5, 7, 6, 9, 5, 6, 7", label="Weekly defects BEFORE the action (comma separated)")
                    post = gr.Textbox("3, 2, 4, 3, 2, 3, 2, 3, 1, 2, 3, 2", label="Weekly defects AFTER the action")
                    t2 = gr.Slider(20, 90, 50, step=5, label="Target reduction (%)")
                    c2 = gr.Slider(80, 99, 90, step=1, label="Confidence needed to close (%)")
                    b2 = gr.Button("Verify effectiveness", variant="primary")
                with gr.Column(scale=3):
                    o2 = gr.HTML()
                    p2 = gr.LinePlot(x="week", y="defects", color="phase", y_title="Defects per week", height=280)
            b2.click(do_verify, [pre, post, t2, c2], [o2, p2])
        with gr.Tab("3 · Benchmark vs plant practice"):
            gr.Markdown("Simulated CAPAs (strong, borderline, weak and no effect) are judged by plant-style fixed windows and by the "
                        "proposed method. **False closure** = closed although the target was not truly met. Synthetic data; assumptions are editable.")
            with gr.Row():
                rb = gr.Slider(1, 30, 6, step=0.5, label="Baseline defects/week"); tb = gr.Slider(20, 90, 50, step=5, label="Target (%)")
                cb = gr.Slider(80, 99, 90, step=1, label="Confidence (%)"); vb = gr.Slider(0.05, 0.8, 0.35, step=0.05, label="Variability (CV)")
            with gr.Row():
                rp = gr.Slider(200, 3000, 1000, step=100, label="Simulated CAPAs per scenario")
                f1 = gr.Number(200000, label="Cost of a false closure (Rs)"); f2 = gr.Number(50000, label="Cost of a missed effect (Rs)")
                f3 = gr.Number(2000, label="Cost per monitoring week (Rs)")
            b3 = gr.Button("Run benchmark", variant="primary")
            o3 = gr.Markdown(); tb3 = gr.Dataframe(interactive=False)
            p3 = gr.BarPlot(x="method", y="value", color="metric", y_title="%", height=320)
            with gr.Accordion("Per-scenario detail", open=False):
                d3 = gr.Dataframe(interactive=False)
            b3.click(do_bench, [rb, tb, cb, vb, rp, f1, f2, f3], [o3, tb3, p3, d3])
