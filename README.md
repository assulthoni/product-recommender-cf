# product-recommender-cf
Collaborative Filtering on Product Recommendation

This Repository contains ML Modeling and Simple Apps to predict Next Customer Purchase.


## Major Tech Stack

- LightFM
- Streamlit

## Algorithms

Collaborative Filtering Hybrid Model. We use collaborative filtering because it is light model that can be run with minimum computing resource.
In addition, collaborative filtering also can perform well in Product Recommendation. In order to search optimum SGD (Stocastic Gradient Descent), we use WASP [research here](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ee5f2862a43e4ddc1e93aac18b70a0555c1ea32d).

We notice with very limited dataset, we can't achieve high performance in recommendation. But in this case, we still got :
0.91 in AUC Score.

So for very beginning of project. It still acceptable.

## Script

In mock the data to train the model using `utils/mock_data.py`. We randomize entire input based on schema that provided.

To know how we model the machine learning, you can go through `model/exploratory_modeling.ipynb` that contains all EDA, Feature Engineering and modeling.

## How to run Web App

1. Create and activate virtual environment using virtualenv package
```
$ virtualenv venv
$ source venv/bin/activate
```
2. Install all dependencies from `requirements.txt`
```
$ pip install -r requirements.txt
```
3. Run streamlit web app
```
$ streamlit run main.py
```
