# Complete Step-by-Step Guide: Deploying ANPR FastAPI on Render

This guide explains how to host your **ANPR FastAPI** application on **[Render.com](https://render.com)** so anyone in the world can call your API URL with your API Key over HTTPS.

---

## 1. Files Prepared for Render

Your repository is already pre-configured with:
1. [**`Dockerfile`**](file:///e:/API_Engines/ANPR/Dockerfile): Production container with Linux OpenCV libraries, Python 3.10, and CPU PyTorch.
2. [**`requirements.txt`**](file:///e:/API_Engines/ANPR/requirements.txt): Optimized dependencies with lightweight CPU PyTorch wheels and headless OpenCV.
3. [**`render.yaml`**](file:///e:/API_Engines/ANPR/render.yaml): Render Blueprint specification.
4. [**`main.py`**](file:///e:/API_Engines/ANPR/main.py): Supports Render's dynamic `$PORT` environment variable.
5. [**`.dockerignore`**](file:///e:/API_Engines/ANPR/.dockerignore): Keeps Docker builds lean by skipping unnecessary files.

---

## 2. Step 1: Push Code to GitHub

Render deploys directly from your Git repository (GitHub or GitLab).

In your terminal / PowerShell:
```bash
git init
git add .
git commit -m "Add ANPR FastAPI with Render deployment setup"
git branch -M main
git remote add origin https://github.com/gaganpal2003-cmyk/ANPR-API-Engine.git
git push -u origin main
```

> **Note on Model Files**: Ensure that `Detection/model/best_plate.pt` and `Detection/model/best_ocr.pt` (~14 MB each, 29 MB total) are committed and pushed to GitHub. GitHub permits files up to 100 MB directly without Git LFS.

---

## 3. Step 2: Create Web Service on Render

1. Go to **[https://dashboard.render.com](https://dashboard.render.com)** and log in (or sign up for free).
2. Click the blue **New +** button in the top right.
3. Select **Web Service**.
4. Choose **Build and deploy from a Git repository**.
5. Select your repository from the list (or paste the repository URL).

---

## 4. Step 3: Configure Web Service Settings

Fill in the settings form:

| Field | Recommended Value | Notes |
|---|---|---|
| **Name** | `anpr-api` *(or your choice)* | This determines your public URL: `https://anpr-api.onrender.com` |
| **Region** | Singapore (or closest to you/India) | Lower latency |
| **Branch** | `main` | Branch to deploy |
| **Runtime** | **Docker** *(Recommended)* | Uses the pre-configured `Dockerfile` (No missing Linux library issues) |
| **Instance Type** | **Starter** ($7/mo) or **Free** | *Starter (512MB-1GB RAM) is recommended for PyTorch models.* |

> **Alternative (Native Python without Docker):**
> If you prefer Native Python instead of Docker:
> - **Runtime:** `Python 3`
> - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
> - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## 5. Step 4: Add Environment Variables in Render

In the Render settings page, scroll down to the **Environment Variables** section and add:

| Key | Value | Description |
|---|---|---|
| `ANPR_API_KEY` | `anpr_live_secret_key_2026` | Secret API key required by clients |
| `ANPR_CONF_THRESHOLD` | `0.15` | Plate detection threshold |
| `ANPR_OCR_CONF_THRESHOLD` | `0.15` | Character OCR threshold |
| `ENVIRONMENT` | `production` | Disables debug reload |

---

## 6. Step 5: Deploy & Get Your Public URL

1. Click **Create Web Service**.
2. Render will build your Docker image and start the FastAPI service.
3. Once the logs show:
   ```
   [API Server] Ready! Serving on device: cpu
   Application startup complete.
   Uvicorn running on http://0.0.0.0:10000
   ```
4. Render will provide your permanent public HTTPS URL at the top left of the dashboard:
   **`https://<your-service-name>.onrender.com`**

---

## 7. How to Use Your Live Cloud API

Once deployed, anyone can access your API across the internet:

### 1. View Swagger UI Docs Live in Browser
Open:
`https://<your-service-name>.onrender.com/docs`

### 2. Client Python Script
Update your [`.env`](file:///e:/Adani_Khavda_project/Application/1_ANPR/Version_03/.env) or client code:
```python
import requests

API_URL = "https://<your-service-name>.onrender.com/api/v1/anpr/predict"
API_KEY = "anpr_live_secret_key_2026"

headers = {"X-API-Key": API_KEY}

with open("truck.jpg", "rb") as f:
    response = requests.post(API_URL, headers=headers, files={"file": f})

print(response.json())
```

### 3. cURL Request
```bash
curl -X POST "https://<your-service-name>.onrender.com/api/v1/anpr/predict" \
     -H "X-API-Key: anpr_live_secret_key_2026" \
     -F "file=@truck.jpg"
```

---

## 8. Important Render Notes & Tips

1. **Free Tier Inactivity Sleep**:
   - On the Free tier, Render automatically puts services to sleep after 15 minutes of inactivity.
   - When a new request arrives, it takes ~40-60 seconds to wake up.
   - On paid Starter plans ($7/month), the service stays awake 24/7 with zero spin-down.
2. **Health Check Path**:
   - Under **Health Check Path** in Render settings, set `/api/v1/health` so Render can monitor your service health.
