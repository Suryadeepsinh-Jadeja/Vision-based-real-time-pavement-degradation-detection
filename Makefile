# RoadScope one-command targets.
#
# The Makefile replaces run.sh, which had drifted from the actual structure
# (the prototype was deleted in Phase 1). Every target is self-contained and
# assumes `make` is run from the repository root.
#
# Python entry points are the same ones documented elsewhere; the Makefile
# only wraps them so a reviewer can type `make demo` and get a running app.

PYTHON ?= python
VENV ?= venv
PY := $(VENV)/bin/$(PYTHON)

.PHONY: help install check check-camera seed dashboard playground test lint clean gc

help:
	@echo "RoadScope targets:"
	@echo "  make install       create the venv and install all extras"
	@echo "  make check         diagnose weights, config, and imports"
	@echo "  make check-camera URL=http://ip:port/video   test a phone camera link"
	@echo "  make seed          populate var/demo.sqlite with a synthetic survey"
	@echo "  make dashboard     launch the condition dashboard"
	@echo "  make playground    launch the detector playground"
	@echo "  make test          run the test suite"
	@echo "  make lint          ruff check + format check"
	@echo "  make clean         remove generated var/ and var/demo.sqlite"
	@echo "  make gc            run git gc (backup bundle first)"

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[yolo,ui,server,geo,dev]"

check:
	$(PY) -m roadscope.agent --check

check-camera:
	@test -n "$(URL)" || (echo "usage: make check-camera URL=http://<phone-ip>:4747/video"; exit 1)
	$(PY) -m roadscope.agent --check --camera-url "$(URL)"

seed:
	$(PY) scripts/seed_demo_data.py --force

dashboard:
	ROADSCOPE_SERVER_DB=var/demo.sqlite $(PY) -m streamlit run scripts/dashboard.py

playground:
	$(PY) -m streamlit run scripts/detector_playground.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

clean:
	rm -rf var/demo.sqlite var/demo.sqlite-* var/demo.sqlite-wal var/demo.sqlite-shm
	rm -rf var/uploads

gc:
	@echo "Back up first (REBUILD_PLAN.md section 17)."
	git bundle create /tmp/roadscope-backup.bundle --all
	git gc --prune=now