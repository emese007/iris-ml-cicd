# Iris MLOps Pipeline – Full Project

This project demonstrates a complete **MLOps workflow** using a simple Iris classification model, deployed through a fully automated CI/CD pipeline.

It includes:

- Model training with **MLflow autologging**
- A **FastAPI** backend exposing `/` and `/predict`
- A **Streamlit** frontend for predictions
- **Dockerized** backend & frontend
- **GitHub Actions CI/CD** pipeline:
  - tests
  - documentation deployment (MkDocs)
  - Docker build & push to Docker Hub
- Deployment to **Azure App Service** (backend + frontend)

---

## Project Structure

```text
backend/
  app/            # FastAPI application
  ml/             # ML training script (MLflow)
  model/          # Saved model .pkl
  tests/          # Pytest tests
  requirements.txt

frontend/
  app.py          # Streamlit UI
  requirements.txt

docs/
  docs/index.md        # MkDocs documentation
  mkdocs.yml

docker-compose.yml
.github/workflows/ci-cd.yml
README.md
```

---

## Installation (Local Setup)

### 1. Clone the repo

```bash
git clone https://github.com/emese007/iris-ml-cicd.git
cd iris-ml-cicd
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
pip install mkdocs
```

---

## Model Training (MLflow)

### Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

MLflow UI → http://127.0.0.1:5000

### Train the model

```bash
python -m backend.ml.train
```

The model is saved here:

```
backend/model/iris_model.pkl
```

---

## Run the Backend (FastAPI)

Start the API:

```bash
uvicorn backend.app.main:app --reload --port 8001
```

API available:

- http://127.0.0.1:8001  
- http://127.0.0.1:8001/docs (Swagger UI)

---

## Run the Frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

Frontend:  
➡ http://127.0.0.1:8501

---

## Docker (Backend + Frontend)

### Build images

```bash
docker build -t mlops-demo-backend:local ./backend
docker build -t mlops-demo-frontend:local ./frontend
```

### Run containers

```bash
# Backend
docker run -p 8001:8001 mlops-demo-backend:local

# Frontend (with backend URL)
docker run -p 8501:8501 \
  -e BACKEND_URL="http://host.docker.internal:8001" \
  mlops-demo-frontend:local
```

---

## CI/CD Pipeline (GitHub Actions)

On every push to **main**, the workflow:

- installs backend dependencies  
- runs tests  
- builds MkDocs documentation  
- deploys docs to GitHub Pages  
- builds backend + frontend Docker images  
- pushes images to Docker Hub  

Workflow file:

```
.github/workflows/ci-cd.yml
```

Secrets (Docker Hub credentials) are configured in:

> GitHub → Settings → Secrets & Variables → Actions

---

## Deployment on Azure (Backend + Frontend)

### Backend

- Hosted as a container on Azure App Service  
- Docker Hub image: `emesehofi/mlops-demo-backend:latest`
- Exposes port **8001**

Backend URLs:

```
https://iris-backend-app-fggsfndcfteqatbm.francecentral-01.azurewebsites.net
https://iris-backend-app-fggsfndcfteqatbm.francecentral-01.azurewebsites.net/docs
```

---

### Frontend

- Hosted as a container on Azure App Service  
- Docker Hub image: `emesehofi/mlops-demo-frontend:latest`
- Exposes port **8501**
- Requires env variable:

```
BACKEND_URL = https://iris-backend-app-fggsfndcfteqatbm.francecentral-01.azurewebsites.net
```

Frontend URL:

```
https://<your-frontend>.azurewebsites.net](https://iris-frontend-app-gtbdgddpgedqdybk.francecentral-01.azurewebsites.net
```

---

## Documentation (MkDocs)

Local preview:

```bash
cd docs
mkdocs serve
```

Deployed online via GitHub Pages at:

```
https://emese007.github.io/iris-ml-cicd/
```

---

## Summary

This project demonstrates a complete modern MLOps workflow:

- model training with tracking
- API serving with FastAPI
- frontend UI with Streamlit
- containerization with Docker
- automated CI/CD
- deployment to Azure Cloud

Perfect blueprint for real-world machine learning operations.
