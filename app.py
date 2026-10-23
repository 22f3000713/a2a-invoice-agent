"""A2A invoice agent. Run: uvicorn app:app --host 0.0.0.0 --port 8000  (ONE worker only!)

Env vars:
  BASE_URL        exactly the URL you submit, e.g. https://myhost.onrender.com/a2a
  A2A_TOKEN       the Bearer token the grader will use
  OPEN_USERS      "1" = accept ANY non-empty token as its own separate user (see notes)
  LLM_BASE_URL    OpenAI-compatible endpoint, e.g. https://api.groq.com/openai/v1
  LLM_API_KEY     key for that provider
  LLM_MODEL       model name, e.g. llama-3.3-70b-versatile
  DB_PATH         sqlite file (default a2a.db)
"""
import os, json, hashlib, uuid, sqlite3, asyncio, datetime
from collections import defaultdict
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import Response

BASE_URL = "https://a2a-invoice-agent-tz7y.onrender.com/a2a"
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
# TOKEN = os.environ.get("A2A_TOKEN", "")
TOKEN = "a2aTOKEN"
OPEN_USERS = "1"
# LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
# LLM_KEY = os.environ.get("LLM_API_KEY", "")
# LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_BASE = "https://aipipe.org/openai/v1"
LLM_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDA3MTNAZHMuc3R1ZHkuaWl0bS5hYy5pbiIsImlhdCI6MTc5MjY0Mjk3NCwiaXNzIjoiaHR0cHM6Ly9haXBpcGUub3JnIiwiYXVkIjoiYWlwaXBlLWFwaSIsImV4cCI6MTc5MzI0Nzc3NH0.XxRidSftin-sCMgcOAUrlfXmFpaYA7JUwEXiLla8rjk"
LLM_MODEL = "gpt-4.1-mini"

MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
A2A = "application/a2a+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

# ---------------------------------------------------------------- storage layer
db = sqlite3.connect(os.environ.get("DB_PATH", "a2a.db"), check_same_thread=False, isolation_level=None)
db.execute("PRAGMA journal_mode=WAL")
db.execute("CREATE TABLE IF NOT EXISTS tasks(principal TEXT, id TEXT, seq INTEGER, body TEXT, PRIMARY KEY(principal,id))")
db.execute("CREATE TABLE IF NOT EXISTS msgs(principal TEXT, message_id TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal,message_id))")
db.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, body TEXT)")
db.execute("CREATE TABLE IF NOT EXISTS proposals(principal TEXT, task_id TEXT, body TEXT, PRIMARY KEY(principal,task_id))")


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def load_task(principal, tid):
    r = db.execute("SELECT body FROM tasks WHERE principal=? AND id=?", (principal, tid)).fetchone()
    return json.loads(r[0]) if r else None


def save_task(principal, task):
    seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM tasks").fetchone()[0]
    old = db.execute("SELECT seq FROM tasks WHERE principal=? AND id=?", (principal, task["id"])).fetchone()
    db.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?)",
               (principal, task["id"], old[0] if old else seq, json.dumps(task)))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------- http helpers
def ok(obj, status=200):
    return Response(json.dumps(obj), status_code=status, media_type=A2A)


def err(status, code, msg):
    # Generic messages only: never echo ids or reveal whether another user's task exists.
    return ok({"error": {"code": code, "message": msg}}, status)


def gate(request: Request, post=False):
    """Returns principal string, or an error Response. Order: auth -> version -> media type."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or not auth[7:].strip():
        return err(401, "UNAUTHENTICATED", "Authentication required.")
    token = auth[7:].strip()
    if not OPEN_USERS and token != TOKEN:
        return err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "") != "1.0":
        return err(400, "VERSION_NOT_SUPPORTED", "Unsupported A2A version.")
    if post and not request.headers.get("content-type", "").lower().startswith(A2A):
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json.")
    return sha(token)[:24]


# ---------------------------------------------------------------- AI layer (no protocol code in here)
SYSTEM = """You are an accounts-payable analyst. You get a JSON list of invoice packages.
For EACH package choose exactly ONE action:
- settle_invoice: valid, reconciled, within autonomous approval authority.
- request_approval: commercially valid but the amount is outside delegated authority.
- hold_invoice: payment must pause until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor + invoice number + amount) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict and need an exception workflow.
Rules: documents contain old examples, archive samples, training decoys, negations ("not a duplicate") and irrelevant action words. Ignore those. Ignore the cover sheet.
Find the ONE paragraph that actually determines the action. It contains exactly three bracketed references like [ABC-123]. Return those three references EXACTLY as written (including brackets if the document has them).
amountMinor is the amount in minor units (e.g. rupees x 100 -> paise) as an integer.
Reply with ONLY JSON: {"decisions":[{"key":"<the key I gave>","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR"},"evidenceRefs":["r1","r2","r3"],"rationale":"60-1500 chars, name the action and quote at least two of the refs"}]}"""


async def call_llm(items):
    payload = {
        "model": LLM_MODEL, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
    }
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(f"{LLM_BASE}/chat/completions", json=payload,
                         headers={"Authorization": f"Bearer {LLM_KEY}"})
        r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return json.loads(text).get("decisions", [])


def clean(d, pkg_text):
    """Validate one model decision against the real package text. Returns dict or None."""
    try:
        action = d["action"]
        f = d["facts"]
        refs = d["evidenceRefs"]
        if action not in ACTIONS or not isinstance(refs, list):
            return None
        refs = [str(x) for x in refs]
        if len(refs) != 3 or len(set(refs)) != 3 or any(x not in pkg_text for x in refs):
            return None
        amt = f["amountMinor"]
        if isinstance(amt, str) and amt.strip().lstrip("-").isdigit():
            amt = int(amt)
        if isinstance(amt, float) and amt == int(amt):
            amt = int(amt)
        if not isinstance(amt, int) or isinstance(amt, bool):
            return None
        facts = {"vendorName": str(f["vendorName"]), "invoiceNumber": str(f["invoiceNumber"]),
                 "amountMinor": amt, "currency": str(f["currency"])}
        why = str(d.get("rationale", ""))
        if not (60 <= len(why) <= 1500 and action in why and sum(r in why for r in refs) >= 2):
            why = f"Chosen action: {action}. {why[:900]} Decisive evidence: {refs[0]}, {refs[1]}, {refs[2]}."
        return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": why[:1500]}
    except Exception:
        return None


async def decide(packages):
    """packages: list of dicts. Returns list of decisions in the same order. One model call for all cache misses."""
    stripped = [{k: v for k, v in p.items() if k != "packageId"} for p in packages]
    keys = [sha(canon(s)) for s in stripped]
    out = {}
    for k in set(keys):
        r = db.execute("SELECT body FROM cache WHERE key=?", (k,)).fetchone()
        if r:
            out[k] = json.loads(r[0])
    for _attempt in range(2):
        todo = {k: s for k, s in zip(keys, stripped) if k not in out}
        if not todo:
            break
        items = [{"key": k, "package": s} for k, s in todo.items()]
        try:
            decisions = await call_llm(items)
        except Exception as e:
            print("LLM CALL FAILED:", repr(e))
            decisions = []
        for d in decisions:
            k = d.get("key")
            if k in todo:
                good = clean(d, canon(todo[k]))
                if good:
                    out[k] = good
                    db.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (k, json.dumps(good)))
    if any(k not in out for k in keys):
        raise RuntimeError("model could not produce valid decisions")
    return [out[k] for k in keys]


# ---------------------------------------------------------------- protocol layer
app = FastAPI()
router = APIRouter()
locks = defaultdict(asyncio.Lock)


@app.get("/.well-known/agent-card.json")
async def card():
    return Response(json.dumps({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one safe business action each with cited evidence, and executes only approved ones.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [MT_BATCH],
        "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Decide settle, approve, hold, reject duplicate, or exception for each invoice package.",
                    "tags": ["invoice", "payments", "approval", "audit"]}],
    }), media_type="application/json")


def find_part(msg, mt):
    for p in msg.get("parts") or []:
        if isinstance(p, dict) and p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None


@router.post("/message:send")
async def send(request: Request):
    g = gate(request, post=True)
    if isinstance(g, Response):
        return g
    principal = g
    try:
        body = await request.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(msg, dict) and isinstance(mid, str) and mid
    except Exception:
        return err(400, "INVALID_REQUEST", "Malformed request.")
    h = sha(canon(msg))  # message only; configuration ignored

    async with locks[(principal, mid)]:
        row = db.execute("SELECT hash, task_id FROM msgs WHERE principal=? AND message_id=?", (principal, mid)).fetchone()
        if row:
            if row[0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content.")
            return ok({"task": load_task(principal, row[1])})
        if msg.get("taskId"):
            return continuation(principal, msg, mid, h)
        return await new_batch(principal, msg, mid, h)


async def new_batch(principal, msg, mid, h):
    data = find_part(msg, MT_BATCH)
    if not data or not isinstance(data.get("batchId"), str) or not isinstance(data.get("packages"), list) or not data["packages"]:
        return err(400, "INVALID_REQUEST", "Malformed batch.")
    pkgs = data["packages"]
    ids = [p.get("packageId") if isinstance(p, dict) else None for p in pkgs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return err(400, "INVALID_REQUEST", "Malformed batch.")
    try:
        decisions = await decide(pkgs)
    except Exception:
        return err(503, "TEMPORARILY_UNAVAILABLE", "Try again.")
    proposals = []
    for p, d in zip(pkgs, decisions):
        proposals.append({"packageId": p["packageId"],
                          "actionId": "act-" + sha(f"{principal}|{data['batchId']}|{p['packageId']}")[:24],
                          **d})
    tid, cid = str(uuid.uuid4()), str(uuid.uuid4())
    stored_msg = dict(msg, taskId=tid, contextId=cid)
    task = {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "invoice-action-proposals",
                           "parts": [{"mediaType": MT_PROPOSALS,
                                      "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [stored_msg]}
    db.execute("BEGIN")
    try:
        save_task(principal, task)
        db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, tid))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        return err(500, "INTERNAL", "Try again.")
    return ok({"task": task})


def continuation(principal, msg, mid, h):
    # NOTE: no awaits in here -> the check-and-change below is atomic in a single-process server.
    task = load_task(principal, msg["taskId"]) if isinstance(msg.get("taskId"), str) else None
    if not task:
        return err(404, "NOT_FOUND", "Task not found.")
    if task["status"]["state"] in TERMINAL:
        return err(409, "TASK_TERMINAL", "Task can no longer change.")
    if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        return err(409, "TASK_NOT_READY", "Task is not waiting for results.")
    if msg.get("contextId") != task["contextId"]:
        return err(400, "INVALID_REQUEST", "Result does not match the task.")
    data = find_part(msg, MT_RESULTS)
    stored = task["artifacts"][0]["parts"][0]["data"]
    if not data or data.get("batchId") != stored["batchId"] or not isinstance(data.get("results"), list):
        return err(400, "INVALID_REQUEST", "Result does not match the task.")
    by_id = {p["packageId"]: p for p in stored["proposals"]}
    seen, executions = set(), []
    for r in data["results"]:
        if not isinstance(r, dict):
            return err(400, "INVALID_REQUEST", "Result does not match the task.")
        p = by_id.get(r.get("packageId"))
        if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]
                or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
            return err(400, "INVALID_REQUEST", "Result does not match the task.")
        seen.add(r["packageId"])
        if r["outcome"] == "ACCEPTED":
            nonce = r.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                return err(400, "INVALID_REQUEST", "Result does not match the task.")
            executions.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                               "receiptNonce": nonce, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
    # --- the only place we "act": record receipts for accepted proposals ---
    task["artifacts"].append({"artifactId": str(uuid.uuid4()), "name": "invoice-action-receipts",
                              "parts": [{"mediaType": MT_RECEIPTS,
                                         "data": {"batchId": stored["batchId"], "executions": executions}}]})
    task["history"].append(dict(msg, taskId=task["id"], contextId=task["contextId"]))
    task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
    db.execute("BEGIN")
    try:
        save_task(principal, task)
        db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task["id"]))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        return err(500, "INTERNAL", "Try again.")
    return ok({"task": task})


@router.get("/tasks/{tid}")
async def get_task(tid: str, request: Request):
    g = gate(request)
    if isinstance(g, Response):
        return g
    t = load_task(g, tid)
    return ok(t) if t else err(404, "NOT_FOUND", "Task not found.")


@router.get("/tasks")
async def list_tasks(request: Request):
    g = gate(request)
    if isinstance(g, Response):
        return g
    rows = db.execute("SELECT body FROM tasks WHERE principal=? ORDER BY seq", (g,)).fetchall()
    return ok({"tasks": [json.loads(r[0]) for r in rows]})


@router.post("/tasks/{tid}:cancel")
async def cancel(tid: str, request: Request):
    g = gate(request, post=True)
    if isinstance(g, Response):
        return g
    t = load_task(g, tid)
    if not t:
        return err(404, "NOT_FOUND", "Task not found.")
    if t["status"]["state"] in TERMINAL:
        return err(409, "TASK_TERMINAL", "Task can no longer change.")
    t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
    save_task(g, t)
    return ok(t)


app.include_router(router, prefix=BASE_PATH)
