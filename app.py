
# app.py - FastAPI serving your scikit-learn Pipeline (StandardScaler + MLPRegressor)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np, os

# Prefer ann_grid.pkl if available (best estimator from GridSearchCV)
MODEL_PATH = "ann_grid.pkl" if os.path.exists("ann_grid.pkl") else "ann_fixed.pkl"
try:
    PIPE = joblib.load(MODEL_PATH)  # Pipeline: StandardScaler + MLPRegressor
except Exception as e:
    raise RuntimeError(f"Could not load model file {MODEL_PATH}: {e}")

FEATURE_ORDER = ["f1","f2","f3","f4","f5"]  # must match training

class Payload(BaseModel):
    features: dict
    metadata: dict | None = None

app = FastAPI(title="SC Life Predictor", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "model_file": MODEL_PATH, "features": FEATURE_ORDER}

@app.post("/predict")
def predict(p: Payload):
    # validate and order features
    try:
        x = np.array([[float(p.features[k]) for k in FEATURE_ORDER]], dtype=float)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"missing feature: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid feature value: {e}")

    # run inference
    try:
        y = float(PIPE.predict(x)[0])  # model predicts cycle life directly (no log transform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

    return {
        "prediction": {"cycle_life": y},
        "model": {"file": MODEL_PATH, "features": FEATURE_ORDER}
    }

# --- INÍCIO DO BLOCO NOVO /ingest_simple ---
from fastapi import Request
import math, numpy as np

# memória simples para acumular a,b por device (apenas para o teste)
ACC = {}  # device_id -> {n,Sx,Sy,Sxx,Sxy}

def _acc(dev):
    return ACC.setdefault(dev, {"n":0,"Sx":0.0,"Sy":0.0,"Sxx":0.0,"Sxy":0.0})

def _features_from_series(samples, i_set_a):
    # samples: lista de dicts {"t_ms":int, "v":float, "i":float}
    samples = sorted(samples, key=lambda s: s["t_ms"])
    def v_at(t_s: float) -> float:
        t_ms = t_s*1000.0
        for k in range(len(samples)-1):
            a, b = samples[k], samples[k+1]
            if a["t_ms"] <= t_ms <= b["t_ms"]:
                t0, v0 = a["t_ms"], a["v"]
                t1, v1 = b["t_ms"], b["v"]
                w = (t_ms - t0)/(t1 - t0) if t1>t0 else 0.0
                return v0 + w*(v1 - v0)
        return samples[-1]["v"]

    V0p = v_at(0.0)
    V10 = v_at(10.0)
    V20 = v_at(20.0)
    f1 = (V0p - V10)   # (IRdrop + ΔV 0–10s) – para teste usamos V0+ diretamente
    f2 = (V10  - V20)  # ΔV 10–20s

    pts = [(s["t_ms"]/1000.0, s["v"]) for s in samples if 5.0 <= s["t_ms"]/1000.0 <= 25.0]
    if len(pts) < 5:
        raise ValueError("Poucos pontos na janela 5–25 s")
    t = np.array([p[0] for p in pts], dtype=float)
    v = np.array([p[1] for p in pts], dtype=float)
    tbar, vbar = t.mean(), v.mean()
    m = ((t - tbar) @ (v - vbar)) / max(((t - tbar) @ (t - tbar)), 1e-12)  # dV/dt (V/s)
    f3 = float(i_set_a / abs(m))  # C @ ciclo

    return f1, f2, f3

@app.post("/ingest_simple")
async def ingest_simple(payload: dict):
    """
    Espera JSON:
    {
      "device_id": "esp32-001",
      "cycle_index": 1,
      "i_set_a": 0.20,
      "samples": [ {"t_ms":0,"v":2.70,"i":0.20}, {"t_ms":1000,"v":2.68,"i":0.20}, ... ]
    }
    """
    try:
        dev  = str(payload["device_id"])
        N    = int(payload["cycle_index"])
        Iset = float(payload["i_set_a"])
        samples = payload["samples"]
        assert isinstance(samples, list) and len(samples) >= 10
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"payload inválido: {e}")

    # calcula f1,f2,f3 a partir das amostras
    try:
        f1, f2, f3 = _features_from_series(samples, Iset)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"erro ao extrair features: {e}")

    # atualiza acumuladores (para f4=a e f5=b)
    A = _acc(dev)
    x = math.log(max(N,1)); y = math.log(max(f3, 1e-12))
    A["n"]  += 1
    A["Sx"] += x
    A["Sy"] += y
    A["Sxx"]+= x*x
    A["Sxy"]+= x*y

    n,Sx,Sy,Sxx,Sxy = A["n"], A["Sx"], A["Sy"], A["Sxx"], A["Sxy"]
    denom = (n*Sxx - Sx*Sx) or 1e-12
    b = (n*Sxy - Sx*Sy) / denom
    ln_a = (Sy - b*Sx)/n
    a = math.exp(ln_a)

    # monta vetor na mesma ordem usada no treino
    X = np.array([[f1, f2, f3, a, b]], dtype=float)
    try:
        y = float(PIPE.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"falha na predição: {e}")

    return {
        "features": {"f1":f1,"f2":f2,"f3":f3,"a":a,"b":b, "cycle_index":N},
        "prediction": {"cycle_life": y},
        "model": {"file": MODEL_PATH, "features": ["f1","f2","f3","f4","f5"]}
    }
# --- FIM DO BLOCO NOVO /ingest_simple ---
