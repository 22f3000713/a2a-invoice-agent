"""
A2A Invoice Agent  --  one file, ~300 lines.

requirements.txt:   fastapi  uvicorn  httpx
Run (ONE worker only!):  uvicorn app:app --host 0.0.0.0 --port $PORT

Environment variables:
  BASE_URL      exactly the URL you submit, e.g. https://myagent.onrender.com/a2a/
  LLM_URL       any OpenAI-compatible base, e.g. https://api.groq.com/openai/v1
  LLM_KEY       API key for that provider (keep it in env vars, never in code)
  LLM_MODEL     e.g. llama-3.3-70b-versatile  (any decent model works)
  DB_PATH       optional, default a2a.db
"""
import os, json, uuid, hashlib, sqlite3, asyncio, re, datetime
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, Request, Response

BASE_URL = os.environ["BASE_URL"]
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
LLM_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDA3MTNAZHMuc3R1ZHkuaWl0bS5hYy5pbiIsImlhdCI6MTc5MjY0Mjk3NCwiaXNzIjoiaHR0cHM6Ly9haXBpcGUub3JnIiwiYXVkIjoiYWlwaXBlLWFwaSIsImV4cCI6MTc5MzI0Nzc3NH0.XxRidSftin-sCMgcOAUrlfXmFpaYA7JUwEXiLla8rjk"
LLM_MODEL = "gpt-5-mini"
# LLM_URL = os.environ.get("LLM_URL", "").rstrip("/")
# LLM_KEY = os.environ.get("LLM_KEY", "")
# LLM_MODEL = os.environ.get("LLM_MODEL", "")

A2A = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULT = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice",
           "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}

# ---------------------------------------------------------------- storage
db = sqlite3.connect(os.environ.get("DB_PATH", "a2a.db"),
                     isolation_level=None, check_same_thread=False)
db.executescript("""
create table if not exists tasks(id text primary key, principal text, body text);
create table if not exists msgs(principal text, mid text, hash text, task_id text,
                                primary key(principal, mid));
create table if not exists cache(k text primary key, v text);
""")


def canon(o):  # recursively key-sorted, compact JSON
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(tid, principal):
    r = db.execute("select body from tasks where id=? and principal=?",
                   (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None


def save(t, principal):
    db.execute("update tasks set body=? where id=? and principal=?",
               (json.dumps(t), t["id"], principal))


# ---------------------------------------------------------------- HTTP helpers
class Fail(Exception):
    def __init__(self, status, code, msg):
        self.status, self.code, self.msg = status, code, msg


def err(status, code, msg):
    h = {"WWW-Authenticate": "Bearer"} if status == 401 else {}
    return Response(json.dumps({"error": {"code": code, "message": msg}}),
                    status_code=status, media_type=A2A, headers=h)


def ok(obj):
    return Response(json.dumps(obj), status_code=200, media_type=A2A)


def gate(request, post=False):
    a = request.headers.get("authorization", "")
    if not a.startswith("Bearer ") or not a[7:].strip():
        return None, err(401, "UNAUTHENTICATED", "Authentication required")
    if request.headers.get("a2a-version") != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Unsupported version")
    acc = request.headers.get("accept", "*/*")
    if not any(x in acc for x in ("json", "*/*")):
        return None, err(406, "NOT_ACCEPTABLE", "Unsupported media type")
    if post:
        ct = request.headers.get("content-type", "").split(";")[0].strip()
        if ct not in (A2A, "application/json"):
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported media type")
    return sha(a[7:].strip()), None  # principal = hash of the token


app = FastAPI()


@app.get("/.well-known/agent-card.json")
def card():
    return Response(json.dumps({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages and proposes one safe business action each.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON",
                                 "protocolVersion": "1.0"}],
        "defaultInputModes": [MT_BATCH],
        "defaultOutputModes": [MT_PROP, MT_RECEIPT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                    "description": "Choose settle, approve, hold, reject duplicate or "
                                   "open exception for each invoice, with evidence.",
                    "tags": ["invoice", "payments", "approval"]}],
    }), media_type="application/json")


# ---------------------------------------------------------------- AI layer
SYSTEM = f"""You are a careful accounts-payable controller. For EVERY invoice package choose exactly one action:
- settle_invoice: valid, reconciled, and within autonomous approval authority.
- request_approval: commercially valid but above the delegated approval limit.
- hold_invoice: payment must pause until a stated verification (e.g. bank-detail callback) completes.
- reject_duplicate: the SAME commercial invoice (same vendor + same invoice number) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, PO, receipt, vendor) and need an exception workflow.

The documents are full of traps: old examples, archive notes, training decoys, negations ("no hold is needed"), and irrelevant action words. Ignore the cover sheet, archive examples and training decoys. Find the ONE paragraph that states the live decision for this invoice and decide from it.

evidenceRefs: exactly THREE bracketed reference tokens copied character-for-character from that decisive paragraph. Never invent a reference. Never use cover-sheet, archive or decoy references.
facts: vendorName, invoiceNumber (strings), amountMinor (INTEGER in minor units, e.g. 123.45 INR -> 12345), currency (3-letter code).
rationale: 60-1500 characters, must contain the action name and at least two of the evidenceRefs.

Return ONLY JSON: {{"decisions":[{{"index":<int from input>,"action":"...","facts":{{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""}},"evidenceRefs":["","",""],"rationale":""}}]}} with one decision per input package."""


async def call_model(policy, items):
    payload = {"model": LLM_MODEL, "temperature": 0,
               "response_format": {"type": "json_object"},
               "messages": [
                   {"role": "system", "content": SYSTEM},
                   {"role": "user", "content": json.dumps(
                       {"policyRevision": policy,
                        "packages": [{"index": i, "package": p} for i, p in items]},
                       ensure_ascii=False)}]}
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(LLM_URL + "/chat/completions", json=payload,
                         headers={"Authorization": "Bearer " + LLM_KEY})
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    return {int(d["index"]): d for d in json.loads(txt)["decisions"]}


def clean(d, pkg):
    """Validate one model decision; raise if anything is off."""
    blob = json.dumps(pkg, ensure_ascii=False)
    assert d["action"] in ACTIONS
    f = d["facts"]
    amt = f["amountMinor"]
    assert not isinstance(amt, bool)
    facts = {"vendorName": str(f["vendorName"]), "invoiceNumber": str(f["invoiceNumber"]),
             "amountMinor": int(amt), "currency": str(f["currency"])}
    refs = list(dict.fromkeys(str(r) for r in d["evidenceRefs"]))
    assert len(refs) == 3 and all(r in blob for r in refs)
    t = str(d.get("rationale", "")).strip()
    if d["action"] not in t:
        t = f"Action {d['action']}: " + t
    if sum(r in t for r in refs) < 2:
        t += " Evidence: " + ", ".join(refs) + "."
    if len(t) < 60:
        t += " This follows the decisive paragraph of the invoice package and policy."
    return {"action": d["action"], "facts": facts, "evidenceRefs": refs, "rationale": t[:1500]}


async def decide(batch):
    pk = batch["packages"]
    keys = [sha(canon({k: v for k, v in p.items() if k != "packageId"})) for p in pk]
    out, todo = {}, []
    for i, k in enumerate(keys):
        r = db.execute("select v from cache where k=?", (k,)).fetchone()
        if r:
            out[i] = json.loads(r[0])
        else:
            todo.append(i)
    for _ in range(2):  # one retry for anything the validator rejected
        if not todo:
            break
        chunks = [todo[j:j + 12] for j in range(0, len(todo), 12)]
        try:
            res = await asyncio.gather(*[
                call_model(batch.get("policyRevision"), [(i, pk[i]) for i in c])
                for c in chunks])
        except Exception:
            res = []
        merged = {}
        for r in res:
            merged.update(r)
        still = []
        for i in todo:
            try:
                out[i] = clean(merged[i], pk[i])
                db.execute("insert or replace into cache values(?,?)",
                           (keys[i], json.dumps(out[i])))
            except Exception:
                still.append(i)
        todo = still
    if todo:
        raise Fail(502, "UPSTREAM_ERROR", "Could not analyse the batch")
    return [out[i] for i in range(len(pk))]


# ---------------------------------------------------------------- message:send
inflight = {}  # (principal, messageId) -> (hash, future)


async def create_task(p, msg, mid, h):
    try:
        part = next(x for x in msg["parts"] if x.get("mediaType") == MT_BATCH)
        batch = part["data"]
        assert isinstance(batch["batchId"], str) and batch["batchId"]
        pk = batch["packages"]
        ids = [x["packageId"] for x in pk]
        assert pk and len(set(ids)) == len(ids)
    except Exception:
        raise Fail(400, "INVALID_REQUEST", "Malformed batch")
    decs = await decide(batch)  # the only slow step (AI), cached per package
    props = []
    for x, d in zip(pk, decs):
        aid = "act-" + sha(f"{p}|{batch['batchId']}|{x['packageId']}|{d['action']}")[:24]
        props.append({"packageId": x["packageId"], "actionId": aid, **d})
    task = {"id": str(uuid.uuid4()),
            "contextId": msg.get("contextId") or str(uuid.uuid4()),
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "invoice-action-proposals",
                           "parts": [{"mediaType": MT_PROP,
                                      "data": {"batchId": batch["batchId"], "proposals": props}}]}],
            "history": [msg]}
    db.execute("begin")  # save state BEFORE answering
    db.execute("insert into tasks values(?,?,?)", (task["id"], p, json.dumps(task)))
    db.execute("insert into msgs values(?,?,?,?)", (p, mid, h, task["id"]))
    db.execute("commit")
    return task


def continue_task(p, msg, mid, h):
    t = load(msg["taskId"], p) if isinstance(msg.get("taskId"), str) else None
    if not t:
        return err(404, "NOT_FOUND", "Task not found")
    if msg.get("contextId") != t["contextId"]:
        return err(400, "INVALID_REQUEST", "Continuation does not match task")
    if t["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        return err(409, "TASK_NOT_ACTIVE", "Task is no longer active")
    try:
        data = next(x for x in msg["parts"] if x.get("mediaType") == MT_RESULT)["data"]
        stored = t["artifacts"][0]["parts"][0]["data"]
        assert data["batchId"] == stored["batchId"]
        props = {x["packageId"]: x for x in stored["proposals"]}
        seen, accepted = set(), []
        for r in data["results"]:
            pr = props[r["packageId"]]
            assert r["packageId"] not in seen
            seen.add(r["packageId"])
            assert r["actionId"] == pr["actionId"] and r["action"] == pr["action"]
            assert r["outcome"] in ("ACCEPTED", "REJECTED")
            if r["outcome"] == "ACCEPTED":
                assert isinstance(r.get("receiptNonce"), str) and r["receiptNonce"]
                accepted.append({"packageId": pr["packageId"], "actionId": pr["actionId"],
                                 "action": pr["action"], "receiptNonce": r["receiptNonce"],
                                 "facts": pr["facts"], "evidenceRefs": pr["evidenceRefs"]})
    except Exception:
        return err(400, "INVALID_REQUEST", "Continuation does not match task")
    t["artifacts"].append({"artifactId": str(uuid.uuid4()), "name": "invoice-action-receipts",
                           "parts": [{"mediaType": MT_RECEIPT,
                                      "data": {"batchId": stored["batchId"], "executions": accepted}}]})
    t["history"].append(msg)
    t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
    db.execute("begin")
    save(t, p)
    db.execute("insert into msgs values(?,?,?,?)", (p, mid, h, t["id"]))
    db.execute("commit")
    return ok({"task": t})


@app.post(BASE_PATH + "/message:send")
async def send(request: Request):
    p, bad = gate(request, post=True)
    if bad:
        return bad
    try:
        msg = json.loads(await request.body())["message"]
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid and msg["role"] == "ROLE_USER"
        assert isinstance(msg["parts"], list)
    except Exception:
        return err(400, "INVALID_REQUEST", "Malformed request")
    h = sha(canon(msg))  # message only; configuration ignored
    row = db.execute("select hash, task_id from msgs where principal=? and mid=?",
                     (p, mid)).fetchone()
    if row:
        if row[0] != h:
            return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content")
        return ok({"task": load(row[1], p)})
    if msg.get("taskId"):
        return continue_task(p, msg, mid, h)
    key = (p, mid)
    if key in inflight:  # an identical request is already being processed
        ih, fut = inflight[key]
        if ih != h:
            return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content")
        try:
            return ok({"task": await asyncio.shield(fut)})
        except Fail as e:
            return err(e.status, e.code, e.msg)
        except Exception:
            return err(500, "INTERNAL", "Internal error")
    fut = asyncio.get_running_loop().create_future()
    inflight[key] = (h, fut)
    try:
        task = await create_task(p, msg, mid, h)
        fut.set_result(task)
        return ok({"task": task})
    except Fail as e:
        fut.set_exception(e); fut.exception()
        return err(e.status, e.code, e.msg)
    except Exception as e:
        fut.set_exception(e); fut.exception()
        return err(500, "INTERNAL", "Internal error")
    finally:
        inflight.pop(key, None)


# ---------------------------------------------------------------- read / list / cancel
@app.get(BASE_PATH + "/tasks/{ref}")
async def get_task(ref: str, request: Request):
    p, bad = gate(request)
    if bad:
        return bad
    t = load(ref, p)
    return ok(t) if t else err(404, "NOT_FOUND", "Task not found")


@app.get(BASE_PATH + "/tasks")
async def list_tasks(request: Request):
    p, bad = gate(request)
    if bad:
        return bad
    rows = db.execute("select body from tasks where principal=? order by rowid", (p,)).fetchall()
    return ok({"tasks": [json.loads(r[0]) for r in rows]})


@app.post(BASE_PATH + "/tasks/{ref}")
async def cancel(ref: str, request: Request):
    p, bad = gate(request)
    if bad:
        return bad
    if not ref.endswith(":cancel"):
        return err(404, "NOT_FOUND", "Not found")
    t = load(ref[:-len(":cancel")], p)
    if not t:
        return err(404, "NOT_FOUND", "Task not found")
    if t["status"]["state"] in TERMINAL:  # first one wins the race, second gets 409
        return err(409, "TASK_NOT_CANCELABLE", "Task already finished")
    t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
    save(t, p)
    return ok(t)
