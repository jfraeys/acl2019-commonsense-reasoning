.PHONY: all env init pdp domain_train clean

# Configuration
DATA=~/acl2019-commonsense-reasoning/data/
PRE_TRAIN_MODEL=bert-base-uncased
FLAGS=--do_lower_case

PYTHON=python3
VENV_DIR=.venv
VENV_PYTHON=$(VENV_DIR)/bin/python
VENV_PIP=$(VENV_DIR)/bin/pip

all: pdp

env:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

init: env

pdp:
	$(VENV_PYTHON) acl2019-commonsense-reasoning/commonsense.py experiments=no_train_base_uncased

domain_train:
	$(VENV_PYTHON) acl2019-commonsense-reasoning/commonsense.py experiments=domain_train_base_uncased

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

