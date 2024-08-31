.PHONY: install
install:
	poetry install

.PHONY: build
build:
	poetry build

.PHONY: test
test:
	poetry run pytest	-v
