# Hugging Face Spaces

This project deploys as two Docker services. Spaces runs one container, so deploy the API and the UI as two Spaces, or run `docker compose up` on a host that can publish both ports.

## API Space

1. Create a Space with the Docker SDK.
2. Push this repository. The image bakes `models/` from the build context, so `make train` has to have been run on the machine that builds the image.
3. Set the Space variables:

```
PORT=7860
SERVICE=api
CORS_ORIGINS=https://<ui-space>.hf.space
RATE_LIMIT=60/minute
```

4. The Dockerfile already honours `PORT` and checks `/health`.

## UI Space

Use the same image and override the command:

```
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

```
SERVICE=ui
PORT=7860
API_BASE_URL=https://<api-space>.hf.space
```

## Local compose

`docker compose up --build` publishes the API on port 8000 and the UI on port 8501. The UI reads `API_BASE_URL=http://api:8000`.

No secret belongs in the image. `.env.example` lists every variable.
