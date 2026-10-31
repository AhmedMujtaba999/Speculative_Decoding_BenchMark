.PHONY: test bench report assets plan clean

test:            ## CPU tests (results checks skip until you benchmark)
	pytest -q

plan:            ## show what `bench` will run, without a GPU
	python bench.py --mode all --dry-run

bench:           ## full GPU sweep: baseline + ngram/draft x gamma 3,5,8
	python bench.py --mode all

report:          ## results/*.json -> results/report.md + charts, copy charts to docs/assets
	python report.py results --assets docs/assets

assets:          ## regenerate README diagrams/theory charts and the demo video
	python scripts/make_readme_assets.py
	python scripts/make_demo_video.py

clean:
	rm -rf results .pytest_cache **/__pycache__
