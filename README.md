
# SC Life Predictor — Deploy (Caminho A)

API REST simples (FastAPI) servindo o seu modelo treinado (Pipeline scikit-learn: StandardScaler + MLPRegressor) a partir de `ann_grid.pkl` (ou `ann_fixed.pkl`).

## 0) Arquivos
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `ann_grid.pkl` (se disponível) e `ann_fixed.pkl` (fallback)
- `requests.http` (testes)
- `example_esp32_snippet.txt` (payload do ESP32)

## 1) Rodando local (sem Docker)
```bash
# Python 3.11+ recomendado
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Coloque ann_grid.pkl e/ou ann_fixed.pkl na mesma pasta (já incluso aqui)
uvicorn app:app --host 0.0.0.0 --port 8080
```

Teste:
```bash
curl -s http://localhost:8080/health | jq
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"f1":0.21,"f2":0.14,"f3":9.7,"f4":11.0,"f5":-0.12}}' | jq
```

## 2) Rodando com Docker (local)
```bash
docker build -t sc-life:1 .
docker run --rm -p 8080:8080 sc-life:1
```

## 3) Publicando na nuvem (exemplos resumidos)

### Opção A — Google Cloud Run (com Dockerfile)
```bash
gcloud builds submit --tag gcr.io/SEU_PROJETO/sc-life:1
gcloud run deploy sc-life --image gcr.io/SEU_PROJETO/sc-life:1 --platform managed --region southamerica-east1 --allow-unauthenticated
```

### Opção B — Render / Railway / Fly.io
- Crie um repositório com estes arquivos.
- Conecte o provedor à sua conta GitHub.
- Crie um novo serviço **Web** apontando para este repo.
- O provedor detecta o Dockerfile automaticamente; a porta é **8080**.

> Segurança básica: use HTTPS no provedor (padrão) e, se necessário, proteja com API key (header `X-API-Key`) — isso exige poucas linhas extras no `app.py`.

## 4) Integração — ESP32 enviando as 5 features
Veja `example_esp32_snippet.txt` (HTTP POST).
No Arduino Cloud, você pode:
- (a) Postar **direto** do ESP32 para `https://SEU_ENDPOINT/predict`, ou
- (b) Usar **Integrations/Webhooks** para postar quando uma variável `features_json` mudar.

## 5) Contrato do payload (JSON)
```json
{
  "features": { "f1": 0.21, "f2": 0.14, "f3": 9.7, "f4": 11.0, "f5": -0.12 },
  "metadata": { "device_id": "esp32-001", "cycle_index": 657 }
}
```

A resposta tem:
```json
{
  "prediction": { "cycle_life": 1234.56 },
  "model": { "file": "ann_grid.pkl", "features": ["f1","f2","f3","f4","f5"] }
}
```

## 6) Dicas
- **Ordem das features** é fixa: `f1..f5` (igual ao treino).
- `ann_grid.pkl` é o melhor (GridSearchCV). Se não existir, a API usa `ann_fixed.pkl`.
- O modelo aqui **já normaliza internamente** (StandardScaler no Pipeline).
