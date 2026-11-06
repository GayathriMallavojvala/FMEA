"""
Case store: SQLite persistence for investigations, tracked actions and an append-only audit log.
Everything the plant would need to answer "who decided what, when, on what evidence".
Path: env CAPA_DB (default ./data/capa.db). Single-file DB, no server needed.
"""
import json
import os
import re
import sqlite3
import time
from datetime import date, timedelta

DB_PATH = os.environ.get("CAPA_DB", os.path.join(os.path.dirname(__file__), "data", "capa.db"))
ACTION_STATUSES = ["Open", "In progress", "Done", "Blocked"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, updated_at TEXT, title TEXT,
  defect_text TEXT, scenario_id TEXT, state_json TEXT, status TEXT DEFAULT 'Investigating',
  capa_text TEXT DEFAULT '', faithfulness_text TEXT DEFAULT '', owner TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS actions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, description TEXT, owner TEXT,
  due_date TEXT, done_when TEXT DEFAULT '', status TEXT DEFAULT 'Open', updated_at TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, ts TEXT, actor TEXT, event TEXT, detail TEXT);
"""


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def connect(path=None):
    path = path or DB_PATH
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def log(con, case_id, event, detail="", actor="system"):
    con.execute("INSERT INTO events(case_id,ts,actor,event,detail) VALUES(?,?,?,?,?)",
                (case_id, _now(), actor, event, detail))


def create_case(con, defect_text, state, owner="", title=None):
    title = title or (defect_text.strip().split(".")[0][:70] or "Untitled case")
    cur = con.execute(
        "INSERT INTO cases(created_at,updated_at,title,defect_text,scenario_id,state_json,owner) VALUES(?,?,?,?,?,?,?)",
        (_now(), _now(), title, defect_text, state["scenario_id"], json.dumps(state), owner))
    log(con, cur.lastrowid, "case_created", f"scenario {state['scenario_id']}", owner or "system")
    con.commit()
    return cur.lastrowid


def save_state(con, case_id, state, event=None, detail="", actor="system"):
    con.execute("UPDATE cases SET state_json=?, updated_at=? WHERE id=?", (json.dumps(state), _now(), case_id))
    if event:
        log(con, case_id, event, detail, actor)
    con.commit()


def save_capa(con, case_id, capa_text, faithfulness_text="", actor="system"):
    con.execute("UPDATE cases SET capa_text=?, faithfulness_text=?, status=CASE WHEN status='Investigating' THEN 'CAPA drafted' ELSE status END, updated_at=? WHERE id=?",
                (capa_text, faithfulness_text, _now(), case_id))
    log(con, case_id, "capa_drafted", f"{len(capa_text)} chars", actor)
    con.commit()


def get_case(con, case_id):
    r = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not r:
        return None
    d = dict(r); d["state"] = json.loads(d.pop("state_json")); return d


def list_cases(con, limit=100):
    return [dict(r) for r in con.execute(
        "SELECT id,created_at,title,scenario_id,status,owner FROM cases ORDER BY id DESC LIMIT ?", (limit,))]


def events(con, case_id):
    return [dict(r) for r in con.execute("SELECT ts,actor,event,detail FROM events WHERE case_id=? ORDER BY id", (case_id,))]


# ---- actions
def parse_action_table(md, today=None):
    """Extract rows from the markdown action table the CAPA prompt asks for:
    | Action | Owner (role) | Due (days from today) | Done when |.  Skips header/separator rows."""
    today = today or date.today()
    out = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.match(r"^\|[\s\-:|]+\|?$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower().startswith("action"):
            continue
        if cells[0].lower() == "what":
            continue
        if len(cells) == 5:
            # Professional action format: What / Where / When / Who / Required Results.
            # Do not turn unapproved generated text into a committed deadline.
            out.append(dict(description=cells[0] + " [Location: " + cells[1] + "]",
                            owner=cells[3], due_date="", done_when=cells[4]))
            continue
        m = re.search(r"\d+", cells[2])
        due = (today + timedelta(days=int(m.group()))).isoformat() if m else ""
        out.append(dict(description=cells[0], owner=cells[1], due_date=due,
                        done_when=cells[3] if len(cells) > 3 else ""))
    return out


def add_actions(con, case_id, actions, actor="system"):
    # Existing execution plans survive regeneration, including IDs and progress.
    if con.execute("SELECT 1 FROM actions WHERE case_id=? LIMIT 1", (case_id,)).fetchone():
        log(con, case_id, "generated_actions_not_imported", "Existing tracked plan retained for review", actor)
        con.commit()
        return 0
    for a in actions:
        con.execute("INSERT INTO actions(case_id,description,owner,due_date,done_when,updated_at) VALUES(?,?,?,?,?,?)",
                    (case_id, a["description"], a.get("owner", ""), a.get("due_date", ""), a.get("done_when", ""), _now()))
    log(con, case_id, "actions_created", f"{len(actions)} actions", actor)
    con.commit()
    return len(actions)


def list_actions(con, case_id=None, include_done=True):
    q, args = "SELECT * FROM actions", []
    conds = []
    if case_id is not None:
        conds.append("case_id=?"); args.append(case_id)
    if not include_done:
        conds.append("status!='Done'")
    if conds:
        q += " WHERE " + " AND ".join(conds)
    return [dict(r) for r in con.execute(q + " ORDER BY due_date, id", args)]


def update_action(con, action_id, status=None, owner=None, due_date=None, actor="user"):
    if status is not None and status not in ACTION_STATUSES:
        raise ValueError(f"status must be one of {ACTION_STATUSES}")
    row = con.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
    if not row:
        raise ValueError(f"no action {action_id}")
    new = dict(status=status or row["status"], owner=owner if owner is not None else row["owner"],
               due_date=due_date if due_date is not None else row["due_date"])
    con.execute("UPDATE actions SET status=?,owner=?,due_date=?,updated_at=? WHERE id=?",
                (new["status"], new["owner"], new["due_date"], _now(), action_id))
    log(con, row["case_id"], "action_updated", f"#{action_id} {row['description'][:40]} -> {new}", actor)
    # close the case automatically when every action is done
    left = con.execute("SELECT COUNT(*) FROM actions WHERE case_id=? AND status!='Done'", (row["case_id"],)).fetchone()[0]
    if left == 0:
        con.execute("UPDATE cases SET status='Actions complete - verify effectiveness', updated_at=? WHERE id=?",
                    (_now(), row["case_id"]))
        log(con, row["case_id"], "all_actions_done", "", "system")
    else:
        con.execute("UPDATE cases SET status='Actions in progress', updated_at=? WHERE id=?",
                    (_now(), row["case_id"]))
    con.commit()


def overdue(con, today=None):
    today = (today or date.today()).isoformat()
    return [dict(r) for r in con.execute(
        "SELECT a.*, c.title FROM actions a JOIN cases c ON c.id=a.case_id "
        "WHERE a.status!='Done' AND a.due_date!='' AND a.due_date<? ORDER BY a.due_date", (today,))]
