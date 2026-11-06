"""Export a saved case as an editable Word document (CAPA record for filing / sign-off)."""
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

FONT = "Arial"
NAVY = RGBColor(0x1F, 0x38, 0x64)


def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _runs(par, text, size=10, bold=False, italic=False, color=None):
    """Write text into a paragraph, honouring **bold** markers from the LLM output."""
    for i, chunk in enumerate(re.split(r"\*\*", text)):
        if not chunk:
            continue
        r = par.add_run(chunk)
        r.font.name = FONT; r.font.size = Pt(size)
        r.bold = bold or (i % 2 == 1); r.italic = italic
        if color is not None:
            r.font.color.rgb = color
    return par


def _heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    _runs(p, text, size=13, bold=True, color=NAVY)
    pPr = p._p.get_or_add_pPr(); b = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "6"), ("space", "1"), ("color", "1F3864")):
        bt.set(qn(f"w:{k}"), v)
    b.append(bt); pPr.append(b)
    return p


def _table(doc, header, rows, widths_cm):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; _shade(c, "1F3864")
        _runs(c.paragraphs[0], h, size=9, bold=True, color=RGBColor(255, 255, 255))
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; _runs(cells[i].paragraphs[0], str(v), size=9)
    for i, w in enumerate(widths_cm):
        t.columns[i].width = Cm(w)
    for row in t.rows:
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
    # repeat header row on page breaks
    trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _page_number_footer(section, left_text):
    p = section.footer.paragraphs[0]; p.text = ""
    _runs(p, left_text + "    Page ", size=8, color=RGBColor(0x59, 0x59, 0x59))
    for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        r = p.add_run(); r.font.size = Pt(8); r.font.name = FONT
        if kind:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._r.append(fc)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)


def _split_capa(text):
    """Return CAPA text without the markdown action table (the tracked table replaces it)."""
    kept = [ln for ln in text.splitlines() if not ln.strip().startswith("|")
            and not re.match(r"^\s*(#+\s*)?(\d\.\s*)?\**\s*action plan\b.*$", ln, re.I)]
    return "\n".join(kept)


def build_case_docx(path, case, scenario, actions, events, belief_rows, evidence, spend):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = Cm(1.8); sec.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    _page_number_footer(sec, f"Case #{case['id']} - DRAFT for qualified engineer review")

    p = doc.add_paragraph(); _runs(p, "Corrective and Preventive Action (CAPA) Record", size=20, bold=True, color=NAVY)
    p = doc.add_paragraph(); _runs(p, case["title"], size=12, color=RGBColor(0x40, 0x40, 0x40))
    p = doc.add_paragraph()
    _runs(p, "DRAFT - generated with AI assistance. Must be reviewed and approved by a qualified engineer before use.",
          size=9, bold=True, color=RGBColor(0xC0, 0x00, 0x00))

    _table(doc, ["Field", "Value"], [
        ["Case number", f"#{case['id']}"], ["Created", case["created_at"]], ["Owner", case["owner"] or "-"],
        ["Status", case["status"]], ["Defect family", scenario["defect_type"].replace("_", " ") if scenario else case["scenario_id"]],
        ["Process stage", scenario["process_stage"].replace("_", " ") if scenario else "-"],
        ["Investigation spend", f"Rs {spend[0]:,} and {spend[1]} min of testing"],
    ], [4.5, 12.5])

    _heading(doc, "Defect description")
    _runs(doc.add_paragraph(), case["defect_text"])

    _heading(doc, "Investigation summary")
    if evidence:
        _runs(doc.add_paragraph(), "Diagnostic tests performed, in order:", bold=True)
        for i, line in enumerate(evidence, 1):
            _runs(doc.add_paragraph(style="List Number"), line)
    else:
        _runs(doc.add_paragraph(), "No diagnostic tests were recorded.", italic=True)
    if belief_rows:
        _runs(doc.add_paragraph(), "Assessed likelihood of each candidate root cause:", bold=True)
        _table(doc, ["Candidate root cause", "Likelihood"], belief_rows, [14.0, 3.0])

    _heading(doc, "CAPA report (AI-drafted)")
    capa = _split_capa(case["capa_text"] or "")
    if capa.strip():
        for line in capa.splitlines():
            if not line.strip():
                continue
            m = re.match(r"^\s*(#+\s*)?(\d\.\s+[^:]{3,60})(:?)\s*(.*)$", line)
            if m and len(m.group(2)) < 60 and not m.group(4):
                q = doc.add_paragraph(); q.paragraph_format.space_before = Pt(8); q.paragraph_format.keep_with_next = True
                _runs(q, m.group(2), bold=True, size=11)
            elif line.lstrip().startswith(("- ", "* ")):
                _runs(doc.add_paragraph(style="List Bullet"), line.lstrip()[2:])
            else:
                _runs(doc.add_paragraph(), re.sub(r"^#+\s*", "", line))
    else:
        _runs(doc.add_paragraph(), "CAPA report not yet drafted.", italic=True)

    _heading(doc, "Action plan and status")
    if actions:
        _table(doc, ["#", "Action", "Owner", "Due", "Status", "Done when"],
               [[a["id"], a["description"], a["owner"], a["due_date"] or "-", a["status"], a["done_when"]] for a in actions],
               [1.0, 5.6, 2.8, 2.2, 2.0, 3.4])
    else:
        _runs(doc.add_paragraph(), "No actions recorded.", italic=True)

    if case.get("faithfulness_text"):
        _heading(doc, "Evidence check of the draft")
        first = case["faithfulness_text"].strip().splitlines()[0]
        _runs(doc.add_paragraph(), first)
        _runs(doc.add_paragraph(), "The check compares each statement in the draft with the retrieved reference material; "
              "it does not verify facts on the plant floor.", size=9, italic=True)

    _heading(doc, "Audit trail")
    _table(doc, ["Time", "By", "Event", "Detail"],
           [[e["ts"], e["actor"], e["event"].replace("_", " "), e["detail"]] for e in events], [3.4, 2.4, 3.4, 7.8])

    _heading(doc, "Review and approval")
    _table(doc, ["Role", "Name", "Signature", "Date"],
           [["Prepared by", "", "", ""], ["Reviewed by", "", "", ""], ["Approved by", "", "", ""]], [3.5, 5.0, 5.5, 3.0])
    for row in doc.tables[-1].rows[1:]:
        row.height = Cm(1.0)
    doc.save(path)
    return path
