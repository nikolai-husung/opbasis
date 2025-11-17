.PHONY: install

VERSION := $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

ifdef OFFLINE
FLAGS = --no-isolation
endif

dist/opbasis-$(VERSION)-py3-none-any.whl: opbasis/*.py
	python3 -m build --wheel $(FLAGS); \
	cd dist; \
	pip3 install --force-reinstall --no-deps opbasis-$(VERSION)-py3-none-any.whl; \
	cd ..

install: dist/opbasis-$(VERSION)-py3-none-any.whl
	cd dist; \
	pip3 install --force-reinstall --no-deps opbasis-$(VERSION)-py3-none-any.whl; \
	cd ..
