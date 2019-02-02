all: flake8 mypy

flake8:
	flake8 --ignore=E501,E303,E402

mypy:
	mypy --ignore-missing-imports .

.PHONY: flake8 mypy all