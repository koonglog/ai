# KoongLog AI Fixed URL Deployment

This setup runs the LightGBM AI API locally on port `8001` and exposes it through a fixed ngrok Domain.

## 1. Prepare ngrok

Create or claim an ngrok static Domain from the ngrok dashboard, for example:

```text
your-ai-domain.ngrok-free.dev
```

Then save the authtoken on this PC:

```powershell
.\deploy\configure-ngrok.ps1
```

## 2. Start the AI model server

Open PowerShell in the repository root:

```powershell
.\deploy\start-ai-server.ps1
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

## 3. Open the fixed URL

In a second PowerShell window:

```powershell
.\deploy\start-ai-ngrok-fixed.ps1 -Domain "your-ai-domain.ngrok-free.dev"
```

External health check:

```powershell
Invoke-RestMethod https://your-ai-domain.ngrok-free.dev/health
```

## 4. Point the backend at the fixed AI URL

When starting the backend, set:

```powershell
$env:AI_SERVICE_URL = "https://your-ai-domain.ngrok-free.dev"
```

Or start the backend with:

```powershell
.\deploy\start-backend-with-ai.ps1 -AiServiceUrl "https://your-ai-domain.ngrok-free.dev"
```

## Notes

- The URL stays fixed, but the local PC must stay on and both the AI server and ngrok process must keep running.
- If the backend is deployed somewhere else, set `AI_SERVICE_URL` in that deployment environment and restart it.
- For cloud production, use a VM/service manager or a hosted platform so the AI server restarts automatically after crashes or reboots.
