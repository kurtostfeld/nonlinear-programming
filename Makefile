.PHONY: flake8
flake8:
	flake8 --extend-ignore E501 *.py

.PHONY: test
test:
	pytest
