# Railway AI Deployment

Use this when deploying only the KoongLog AI model API as a Railway service.

## Required variables

Set these in the Railway service variables:

```text
AI_CLASSIFIER_BACKEND=lightgbm
AI_LGBM_MODEL_DIR=./ai/artifacts/models_pattern_2026_05_11_18_no_accel
AI_LGBM_MIN_CONFIDENCE=0.6
AI_LGBM_SHADOW_MODE=false
ENABLE_OPENAI=false
```

Optional, only when LLM message generation is needed:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-mini
ENABLE_OPENAI=true
```

## Railway settings

The repository includes `railway.json`, which tells Railway to:

- run `python -m ai.serve`
- use `/health` as the healthcheck path
- restart on failure

The repository also includes `railpack.json`, which installs `libgomp1` in the
runtime image for LightGBM.

After deployment, generate a Railway domain from:

```text
Service -> Settings -> Networking -> Public Networking -> Generate Domain
```

## Backend connection

Set the backend service variable to the AI Railway URL:

```text
AI_SERVICE_URL=https://your-ai-service.up.railway.app
```

Then redeploy or restart the backend service.

## Smoke test

```powershell
Invoke-RestMethod https://your-ai-service.up.railway.app/health
```

Expected response contains:

```json
{
  "status": "ok",
  "classifier_backend": "lightgbm"
}
```

The dashboard DB part may show `unavailable` when only the AI service is deployed. That is fine for the backend integration routes such as `/api/v1/ai/classify-event`.
