.PHONY: clean test coverage

help:
	@echo "    test"
	@echo "        Run uniitest"
	@echo "    clean"
	@echo "        Clean all ipython cache files "

clean:
	find . -name __pycache__ -type d | xargs rm -rf

test:
	PYTHONPATH=. pytest tests/

