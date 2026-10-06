.PHONY: install run test bench lint clean

install:
	pip install -r requirements.txt
	cp -n .env.example .env || true

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

bench:
	python scripts/benchmark.py --hours 2 --seed 42

clean:
	rm -rf data .pytest_cache **/__pycache__
