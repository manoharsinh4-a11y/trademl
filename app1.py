from __future__ import annotations
from flask import Flask, request, jsonify, send_file, abort
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Tuple
import os, csv, uuid, glob, math, io, zipfile, threading

app = Flask(__name__)

# =============================
# Config & Dirs
# =============================
PORT = int(os.environ.get("PORT", 5000))  # set $env:PORT to override
DATA_DIR = os.environ.get("DATA_DIR", "data_signals")
LABELS_DIR = os.environ.get("LABELS_DIR", "labels")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
API_TOKEN = os.environ.get("API_TOKEN")  # optional shared secret

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Decision thresholds for model inference
THRESH_BUY  = float(os.environ.get("ML_THRESH_BUY", 0.60))
THRESH_SELL = float(os.environ.get("ML_THRESH_SELL", 0.60))

# In-memory state
_last_signal: Dict[str, Any] = {}
_signals: deque = deque(maxlen=2000)

# Model state
_model = None
_model_meta: Dict[str, Any] = {}

# Locks for safe writes
_csv_lock = threading.Lock()
_labels_lock = threading.Lock()

CSV_HEADER = [
    "signal_id","received_utc","source","symbol","exchange","interval","time",
    "price","signal","rsi","macd","atr","ema200","supertrend_dir","volume"
]

# =============================
# Utils
# =============================

def _require_token():
    if not API_TOKEN:
        return
    tok = request.headers.get("X-API-TOKEN") or request.args.get("token")
    if tok != API_TOKEN:
        abort(401)

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _csv_today_path() -> str:
    return os.path.join(DATA_DIR, datetime.utcnow().strftime("%Y-%m-%d") + ".csv")

def _append_csv(row: Dict[str, Any]) -> None:
    path = _csv_today_path()
    write_header = not os.path.isfile(path)
    with _csv_lock, open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_HEADER})

def _labels_path() -> str:
    p = os.path.join(LABELS_DIR, "labels.csv")
    if not os.path.exists(p):
        with open(p, "w", newline="") as f:
            csv.writer(f).writerow(["signal_id","target","pnl","labeled_at"])
    return p

def _num(x, d=None):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return d
        return float(x)
    except Exception:
        return d

def _interval_to_number(x):
    if x is None: return None
    s = str(x).strip().lower()
    for suf in ("minute","minutes","min","m"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    try:
        return float(s)
    except Exception:
        return None

def normalize_features(f: Dict[str, Any]) -> Dict[str, Any]:
    f = f or {}
    st = f.get("supertrend_dir")
    st = (str(st).lower() if st is not None else None)
    if st in ("1","up","true","bull","long"):
        st = "up"
    elif st in ("-1","down","false","bear","short"):
        st = "down"
    else:
        st = None
    return {
        "rsi": _num(f.get("rsi")),
        "macd": _num(f.get("macd")),
        "atr": _num(f.get("atr")),
        "ema200": _num(f.get("ema200")),
        "volume": _num(f.get("volume")),
        "supertrend_dir": st,
    }

def _intent_from_signal(sig: str) -> str:
    s = (sig or "WAIT").upper()
    if s in ("LONG","BUY"): return "BUY"
    if s in ("SHORT","SELL"): return "SELL"
    return "WAIT"

# =============================
# Confluence (fallback rules)
# =============================

def ml_decide_confluence(signal: str, price, features: Dict[str, Any]) -> Dict[str, Any]:
    intent = _intent_from_signal(signal)
    rsi = features.get("rsi"); macd = features.get("macd"); st = features.get("supertrend_dir")
    ema = features.get("ema200"); p = _num(price)
    reasons = [f"tv_intent={intent}"]
    action = "WAIT"
    if intent == "BUY":
        if (rsi is not None and rsi>52) and (macd is not None and macd>0) and st=="up":
            action = "BUY"; reasons.append("rsi>52 & macd>0 & trend=up")
        elif (p is not None and ema is not None and p>ema) and st=="up":
            action = "BUY"; reasons.append("price>ema200 & trend=up")
    elif intent == "SELL":
        if (rsi is not None and rsi<48) and (macd is not None and macd<0) and st=="down":
            action = "SELL"; reasons.append("rsi<48 & macd<0 & trend=down")
        elif (p is not None and ema is not None and p<ema) and st=="down":
            action = "SELL"; reasons.append("price<ema200 & trend=down")
    conf = 0.5 if action != "WAIT" else 0.2
    return {"action": action, "confidence": conf, "reason": "; ".join(reasons)}

# =============================
# ML model utils (train/load/predict)
# =============================

def _model_path() -> str:
    return os.path.join(MODELS_DIR, "model.pkl")

def _meta_path() -> str:
    return os.path.join(MODELS_DIR, "model_meta.csv")

def load_model() -> None:
    global _model, _model_meta
    try:
        from joblib import load
        if os.path.exists(_model_path()):
            _model = load(_model_path())
            meta: Dict[str, Any] = {}
            if os.path.exists(_meta_path()):
                with open(_meta_path(), "r") as f:
                    r = csv.reader(f)
                    for row in r:
                        if not row:
                            continue
                        meta[row[0]] = row[1] if len(row)>1 else ""
            _model_meta = meta
        else:
            _model = None; _model_meta = {}
    except Exception:
        _model = None; _model_meta = {}

def _feature_vector(entry: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    inp = entry["input"]; feat = inp["features"]
    price = _num(inp.get("price")); ema = feat.get("ema200")
    super_up = 1.0 if feat.get("supertrend_dir")=="up" else 0.0 if feat.get("supertrend_dir")=="down" else math.nan
    price_above_ema = 1.0 if (price is not None and ema is not None and price>ema) else (0.0 if (price is not None and ema is not None) else math.nan)
    sym = str(inp.get("symbol",""))
    is_banknifty = 1.0 if "BANKNIFTY" in sym.upper() else 0.0
    intent = _intent_from_signal(inp.get("signal")); is_long_intent = 1.0 if intent=="BUY" else 0.0
    interval = _num(inp.get("interval"))
    vol = feat.get("volume")

    names = [
        "rsi","macd","atr","ema200","volume",
        "supertrend_up","price_above_ema","is_banknifty","is_long_intent","interval",
    ]
    x = [feat.get("rsi"), feat.get("macd"), feat.get("atr"), feat.get("ema200"), vol,
         super_up, price_above_ema, is_banknifty, is_long_intent, interval]
    x = [0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else float(v) for v in x]
    return x, names

def predict_with_model(entry: Dict[str, Any]) -> Dict[str, Any]:
    global _model
    if _model is None:
        return {"available": False}
    try:
        x, _ = _feature_vector(entry)
        proba = float(_model.predict_proba([x])[0][1])
    except Exception:
        return {"available": False}
    intent = _intent_from_signal(entry["input"]["signal"]) 
    action = "WAIT"
    if intent == "BUY" and proba >= THRESH_BUY:
        action = "BUY"
    elif intent == "SELL" and (1.0 - proba) >= THRESH_SELL:
        action = "SELL"
    conf = proba if intent=="BUY" else (1.0 - proba) if intent=="SELL" else 0.5
    return {"available": True, "proba_success": proba, "action": action, "confidence": round(conf,3),
            "reason": f"model(p_success={proba:.3f}, intent={intent}, thr_buy={THRESH_BUY}, thr_sell={THRESH_SELL})"}

def train_model() -> Dict[str, Any]:
    # Load signals
    rows: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        with open(path, "r") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    # Load labels
    labels_map: Dict[str, Dict[str, Any]] = {}
    lp = _labels_path()
    with open(lp, "r") as f:
        for r in csv.DictReader(f):
            labels_map[r["signal_id"]] = {"target": int(r["target"]), "pnl": float(r.get("pnl") or 0.0)}
    # Join to X/y
    X: List[List[float]] = []; y: List[int] = []
    for r in rows:
        sid = r["signal_id"]
        if sid not in labels_map:
            continue
        entry = {"input": {
            "symbol": r.get("symbol"),
            "exchange": r.get("exchange"),
            "interval": _interval_to_number(r.get("interval")),
            "time": r.get("time"),
            "price": _num(r.get("price")),
            "signal": r.get("signal"),
            "features": normalize_features({
                "rsi": _num(r.get("rsi")),
                "macd": _num(r.get("macd")),
                "atr": _num(r.get("atr")),
                "ema200": _num(r.get("ema200")),
                "supertrend_dir": r.get("supertrend_dir"),
                "volume": _num(r.get("volume")),
            })
        }}
        x_vec, names = _feature_vector(entry)
        X.append(x_vec); y.append(labels_map[sid]["target"])
    if len(X) < 50:
        return {"trained": False, "reason": f"not enough labeled samples: {len(X)} (need ~50+)"}
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from joblib import dump
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    dump(clf, _model_path())
    with open(_meta_path(), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trained_at", _now_iso()])
        w.writerow(["features", ",".join(names)])
        w.writerow(["test_accuracy", f"{acc:.3f}"])
        w.writerow(["samples", str(len(X))])
    load_model()
    return {"trained": True, "test_accuracy": round(float(acc),3), "samples": len(X)}

# =============================
# Routes
# =============================

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": _now_iso(), "model_loaded": _model is not None,
                    "thresholds": {"buy": THRESH_BUY, "sell": THRESH_SELL}, "meta": _model_meta})

@app.route("/metrics")
def metrics():
    today_path = _csv_today_path(); today_rows = 0
    if os.path.exists(today_path):
        with open(today_path, "r") as f:
            today_rows = sum(1 for _ in f) - 1 if os.path.getsize(today_path) > 0 else 0
    return jsonify({"ok": True, "in_memory_signals": len(_signals), "csv_today_rows": max(0, today_rows),
                    "model_loaded": _model is not None})

@app.route("/ml/decide", methods=["POST"])  # stateless decision
def ml_decide():
    _require_token()
    payload = request.get_json(force=True, silent=True) or {}
    features = normalize_features(payload.get("features"))
    info = {
        "source": payload.get("source", "manual"),
        "symbol": (payload.get("symbol") or "").upper(),
        "exchange": payload.get("exchange"),
        "interval": _interval_to_number(payload.get("interval")),
        "time": payload.get("time"),
        "price": payload.get("price"),
        "signal": (payload.get("signal") or "WAIT").upper(),
        "features": features,
    }
    model_dec = predict_with_model({"input": info})
    if model_dec.get("available"):
        decision = {"action": model_dec["action"], "confidence": model_dec["confidence"], "reason": model_dec["reason"]}
    else:
        decision = ml_decide_confluence(signal=info["signal"], price=info["price"], features=features)
    return jsonify({"ok": True, "decision": decision, "input": info})

@app.route("/webhook", methods=["POST"])  # logs to CSV & memory
@app.route("/webhook/tradingview", methods=["POST"])  # alias
def webhook():
    _require_token()
    payload = request.get_json(force=True, silent=True) or {}
    features = normalize_features(payload.get("features"))
    info = {
        "source": payload.get("source", "unknown"),
        "symbol": (payload.get("symbol") or "").upper(),
        "exchange": payload.get("exchange"),
        "interval": _interval_to_number(payload.get("interval")),
        "time": payload.get("time"),
        "price": payload.get("price"),
        "signal": (payload.get("signal") or "WAIT").upper(),
        "features": features,
    }
    entry: Dict[str, Any] = {"signal_id": str(uuid.uuid4()), "received": _now_iso(), "input": info}
    model_dec = predict_with_model({"input": info})
    decision = ( {"action": model_dec["action"], "confidence": model_dec["confidence"], "reason": model_dec["reason"]}
                 if model_dec.get("available") else ml_decide_confluence(info["signal"], info["price"], features) )
    executed = decision["action"] if decision["action"] in ("BUY","SELL") else "NONE"
    entry["ml_decision"] = decision; entry["executed"] = executed
    global _last_signal, _signals
    _last_signal = entry; _signals.appendleft(entry)
    _append_csv({
        "signal_id": entry["signal_id"], "received_utc": entry["received"], "source": info["source"],
        "symbol": info["symbol"], "exchange": info["exchange"], "interval": info["interval"], "time": info["time"],
        "price": info["price"], "signal": info["signal"], "rsi": features.get("rsi"), "macd": features.get("macd"),
        "atr": features.get("atr"), "ema200": features.get("ema200"), "supertrend_dir": features.get("supertrend_dir"),
        "volume": features.get("volume")
    })
    return jsonify({"ok": True, **entry})

@app.route("/last")
def last_signal():
    return jsonify({"ok": True, "last": _last_signal or None})

@app.route("/signals")
def list_signals():
    try:
        limit = int(request.args.get("limit", 50)); limit = max(1, min(limit, len(_signals)))
    except Exception:
        limit = min(50, len(_signals))
    return jsonify({"ok": True, "count": limit, "items": list(_signals)[:limit]})

@app.route("/position")
def position():
    if not _last_signal:
        return jsonify({"ok": True, "position": "FLAT"})
    executed = _last_signal.get("executed", "NONE")
    pos = "LONG" if executed=="BUY" else ("SHORT" if executed=="SELL" else "FLAT")
    return jsonify({"ok": True, "position": pos, "last_signal": _last_signal})

@app.route("/labels", methods=["POST"])  # body: { signal_id, target(0/1), pnl(optional) }
def add_label():
    _require_token()
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("signal_id"); target = payload.get("target"); pnl = payload.get("pnl", 0)
    if not sid:
        return jsonify({"ok": False, "error": "signal_id required"}), 400
    if target not in (0, 1, "0", "1"):
        return jsonify({"ok": False, "error": "target must be 0 or 1"}), 400
    with _labels_lock, open(_labels_path(), "a", newline="") as f:
        w = csv.writer(f); w.writerow([sid, int(target), float(pnl or 0), _now_iso()])
    return jsonify({"ok": True, "labeled": sid}), 201

@app.route("/train", methods=["POST"])  # train from signals+labels
def train_route():
    _require_token()
    try:
        out = train_model(); return jsonify({"ok": out.get("trained", False), **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/model")
def model_meta():
    exists = os.path.exists(_model_path())
    return jsonify({"ok": True, "exists": exists, "loaded": _model is not None, "meta": _model_meta,
                    "path": _model_path() if exists else None})

@app.route("/model/reload", methods=["POST"])  # force reload
def model_reload():
    _require_token(); load_model(); return jsonify({"ok": True, "loaded": _model is not None, "meta": _model_meta})

@app.route("/export/today.csv")
def export_today_csv():
    _require_token(); path = _csv_today_path()
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "no data for today"}), 404
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(path))

@app.route("/export/all.zip")
def export_all_zip():
    _require_token(); mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
            zf.write(path, arcname=os.path.basename(path))
        labels = _labels_path()
        if os.path.exists(labels):
            zf.write(labels, arcname=os.path.basename(labels))
    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="signals_and_labels.zip")

# =============================
# Bootstrap
# =============================
load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)

