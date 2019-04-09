all: flake8 mypy bandit diatra safety

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

.PHONY: flake8 mypy all bandit diatra safety