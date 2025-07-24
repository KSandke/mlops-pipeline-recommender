.PHONY: all clean data preprocess train evaluate serve

all: data train evaluate

clean:
	rm -rf Recommender/data/processed/*
	rm -rf Recommender/models/*

data:
	python Recommender/src/explore_data.py

preprocess:
	python Recommender/src/preprocess_data.py

train:
	python Recommender/src/train.py

evaluate:
	python Recommender/src/evaluator.py

serve:
	uvicorn Recommender.src.api.main:app --host 0.0.0.0 --port 8000 --reload 