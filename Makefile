CODE = src

switch_to_macos:
	rm poetry.lock
	cat utils/pyproject_macos.txt > pyproject.toml

switch_to_linux:
	rm poetry.lock
	cat utils/pyproject_linux.txt > pyproject.toml

install:
	python3.10 -m pip install poetry
	poetry install

lint:
	poetry run pflake8 $(CODE)

format:
	#format code
	poetry run black $(CODE)

download_checkpoints:
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13uT2qFxBZmodJ4VMbhKVWELioEnuDmnU" -O default_test_model/s1/model_best.pth && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lYFMcaDeoPyv1efb8V8uvNw-FYjn23V5" -O default_test_model/s3/model_best.pth && rm -rf /tmp/cookies.txt

train:
	poetry run python train.py -c src/config.json

test_s1:
	poetry run python test.py -r default_test_model/s1/model_best.pth

test_s3:
	poetry run python test.py -r default_test_model/s3/model_best.pth