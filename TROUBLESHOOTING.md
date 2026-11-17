# Troubleshooting – Traffic Sign Recognizer

## 1. Backend Worker Timeout
Problem:
Gunicorn workers experienced timeout and SIGKILL errors.

Observation:
The backend logs showed worker timeouts and possible memory pressure.

Resolution:
Restarted the backend container and verified that the model loaded successfully.

## 2. CUDA Initialization Error
Problem:
TensorFlow reported a CUDA initialization error.

Observation:
The model successfully loaded despite the CUDA warning.

Resolution:
The application continued running using the available CPU resources.

## 3. Frontend Connectivity
Problem:
Frontend needs to communicate with the backend API.

Verification:
Executed the following command:

podman exec traffic-frontend wget -qO- http://backend:5000/health

Result:
{"model_loaded":true,"status":"ok"}

## 4. Container Restart
Command:
podman restart traffic-backend traffic-frontend

Verification:
podman ps

Both containers were confirmed running.

## 5. Persistent Storage
Command:
podman volume inspect sign-db

The database volume was verified.

## 6. Image Registry Verification
Verified backend and frontend images on Docker Hub using:

podman search --list-tags docker.io/shresth11/traffic-sign-backend

podman search --list-tags docker.io/shresth11/traffic-sign-frontend
