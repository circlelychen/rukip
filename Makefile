.PHONY: clean test publish build

help:
	@echo "    build"
	@echo "        Build  source archive and wheel "
	@echo "    publish"
	@echo "        Puclish package "
	@echo "    test"
	@echo "        Run uniitest"
	@echo "    clean"
	@echo "        Clean all ipython cache files "

build:
	python setup.py sdist bdist_wheel

publish:
	python3 -m twine upload dist/*

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	find . -name '__pycache__' -type d -exec rm -rf  {} +
	find . -name '.pytest_cache' -type d -exec rm -rf  {} +
	rm -rf build/
	rm -rf dist/

test:
	pytest rukip --cov rukip
