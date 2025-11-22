# Iris MLOps – Project Documentation

This project demonstrates a complete **MLOps pipeline** built around a simple Iris classification model.

It includes:

- Model training and experiment tracking with **MLflow**
- A **FastAPI** backend exposing a `/predict` endpoint
- A **Streamlit** frontend UI for predictions
- **Docker** images for backend & frontend
- A **CI/CD pipeline** with GitHub Actions
- Deployment to **Azure App Service**

---

## 1. Project Structure

Main folders:

```
backend/
  app/               FastAPI application (API)
  ml/                Model training script with MLflow
  model/             Trained model file (iris_model.pkl)
  tests/             Backend tests (pytest)
  requirements.txt   Backend dependencies

frontend/
  app.py             Streamlit app
  requirements.txt   Frontend dependencies

docs/
  index.md           This documentation (MkDocs)

.github/workflows/
  ci-cd.yml          CI/CD pipeline (tests, docs, Docker build & push)
```

---

## 2. How to Get Started

This section explains how to set up your environment and install all required dependencies.

### 2.1 Prerequisites

You need:

- Python **3.11+**
- A virtual environment
- pip
- Docker (optional)
- Git

Create and activate your environment:

```
cd iris-ml-cicd
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
pip install mkdocs mkdocs-material
```

---

## 3. MLflow & Model Training


### 3.1 Start MLflow UI

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

MLflow UI is available at:  
http://127.0.0.1:5000

### 3.2 Train the Model

The training script:

- loads the Iris dataset  
- trains a LogisticRegression model  
- logs parameters & metrics to MLflow  
- saves the trained model to `backend/model/iris_model.pkl`

Run:

```
python -m backend.ml.train
```

After training, the file `iris_model.pkl` should appear inside:

```
backend/model/
```

---

## 4. Run the Backend (FastAPI)

The backend exposes:

- `GET /` → health check  
- `POST /predict` → returns the predicted Iris class  

Run locally:

```
uvicorn backend.app.main:app --reload --port 8001
```

Backend available at:  
http://127.0.0.1:8001  
http://127.0.0.1:8001/docs

---

## 5. Run the Frontend (Streamlit)

The frontend:

- lets you input the 4 Iris features  
- sends a request to `/predict`  
- displays the predicted class  

Default backend URL:

```
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
```

Run:

```
cd frontend
streamlit run app.py
```

Frontend available at:  
http://127.0.0.1:8501

---

## 6. Run Everything with Docker


### 6.1 Build Docker images

```
docker build -t mlops-demo-backend:local ./backend
docker build -t mlops-demo-frontend:local ./frontend
```

### 6.2 Run containers

**Backend:**

```
docker run -p 8001:8001 mlops-demo-backend:local
```

**Frontend (with backend connection):**

```
docker run -p 8501:8501 \
  -e BACKEND_URL="http://host.docker.internal:8001" \
  mlops-demo-frontend:local
```

---

## 7. CI/CD Pipeline (GitHub Actions)

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) runs on each push to `main`:

- installs backend dependencies  
- runs backend tests  
- builds and deploys documentation  
- builds Docker images (backend + frontend)  
- pushes images to Docker Hub  

Secrets required:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Configured in:

> GitHub → Settings → Secrets and Variables → Actions

---

## 8. Deployment on Azure

### Backend

- Runs as a Docker App Service  
- Port: **8001**  
- Image:

```
DOCKERHUB_USER/mlops-demo-backend:latest
```

Accessible at:

```
https://iris-backend-app-fggsfndcfteqatbm.francecentral-01.azurewebsites.net
https://iris-backend-app-fggsfndcfteqatbm.francecentral-01.azurewebsites.net/docs
```

### Frontend

- Runs as a Docker App Service  
- Port: **8501**  
- Requires environment variable:

```
BACKEND_URL = https://iris-backend-app-fggsfndcfteqatbm.francecentral-01.azurewebsites.net
```

Accessible at:

```
https://iris-frontend-app-gtbdgddpgedqdybk.francecentral-01.azurewebsites.net
```

---

## 9. Summary

This project demonstrates a complete MLOps workflow:

- ML training and experiment tracking  
- FastAPI model serving  
- Streamlit user interface  
- Dockerization  
- CI/CD automation with GitHub Actions  
- Deployment to Azure App Service  

---
