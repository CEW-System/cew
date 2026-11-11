.PHONY: install test data train eval demo benchmark clean logs

install:
	pip install -r requirements.txt

logs:
	mkdir -p logs runs checkpoints data/synthetic

data: logs
	python scripts/generate_data.py

train: logs
	make -C training all

eval:
	python scripts/evaluate.py

demo:
	streamlit run dashboard/app.py

benchmark:
	python scripts/benchmark_latency.py

test:
	pytest -v tests/

clean:
	rm -rf data/synthetic/* logs/* runs/* checkpoints/* .pytest_cache