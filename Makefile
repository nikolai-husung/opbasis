.PHONY: install

VERSION := $(shell grep -m 1 version pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

dist/opbasis-$(VERSION)-py3-none-any.whl: opBasis/*.py
	python3 -m build; \
	cd dist; \
	pip3 install --force-reinstall --no-deps opbasis-$(VERSION)-py3-none-any.whl; \
	cd ..

install: dist/opbasis-$(VERSION)-py3-none-any.whl
	cd dist; \
	pip3 install --force-reinstall --no-deps opbasis-$(VERSION)-py3-none-any.whl; \
	cd ..
