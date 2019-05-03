all: flake8 mypy bandit diatra safety vulture

flake8:
	flake8 --ignore=E501,E303,E402,E252

mypy:
	mypy --ignore-missing-imports .

bandit:
	bandit -r .

diatra:
	python3 -m pydiatra .

safety:
	safety check

vulture:
	vulture .

.PHONY: flake8 mypy all bandit diatra safety vulture