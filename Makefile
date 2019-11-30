.PHONY: clean test coverage

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
	twine upload dist/*

clean:
	find . -name __pycache__ -type d | xargs rm -rf

test:
	PYTHONPATH=. pytest tests/

