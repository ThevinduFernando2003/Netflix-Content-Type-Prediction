PYTHON ?= python
export PYTHONPATH := .

.PHONY: setup profile audit train evaluate test lint format api ui run docker-build docker-up clean

setup:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

profile:
	$(PYTHON) scripts/profile_data.py

audit:
	$(PYTHON) scripts/leakage_audit.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTHON) -m pytest --cov=src --cov-fail-under=80

lint:
	$(PYTHON) -m ruff check src tests app scripts
	$(PYTHON) -m black --check src tests app scripts
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m black src tests app scripts
	$(PYTHON) -m ruff check --fix src tests app scripts

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

run:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t netflix-type-classifier .

docker-up:
	docker compose up --build

clean:
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree('models', ignore_errors=True); pathlib.Path('models').mkdir(); pathlib.Path('models/.gitkeep').write_text('')"
