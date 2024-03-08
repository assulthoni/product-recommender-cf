import pickle
import pandas as pd
from lightfm.data import Dataset


class Recommender():
    def __init__(self) -> None:
        self.model = None
        self.map_user_id = None
        self.map_item_id = None
        self.user_features = None
        self.item_features = None
        self.interactions = None
        self.weights = None
        self.item_ids = None
        self.id_item_features = None

    def preprocess(self, df_user, df_items, df_interaction):
        self.map_user_id_int(df_user)
        self.map_item_id_int(df_items)
        self.df_user = df_user
        self.df_items = df_items
        self.df_interaction = df_interaction

        df_items['product_id_int'] = df_items['product_id'].map(self.map_item_id)
        df_user['customer_id_int'] = df_user['customer_id'].map(self.map_user_id)
        df_interaction['customer_id_int'] = df_interaction['customer_id'].map(self.map_user_id)
        df_interaction['product_id_int'] = df_interaction['product_id'].map(self.map_item_id)

        user_features = df_user[['time_spent', 'page_views']].apply(
                lambda x: ','.join(x.map(str)), axis=1)
        user_features = user_features.str.split(',')
        user_features = user_features.apply(pd.Series).stack().reset_index(drop=True)
        user_ids = set(df_user['customer_id_int'].tolist())
        id_user_features = df_user[['customer_id_int', 'page_views', 'time_spent']].apply(
            lambda row: (row['customer_id_int'], [str(row['page_views']), str(row['time_spent'])]), axis=1
        )

        df_items['price'] = df_items['price'].apply(lambda x: int(x))
        df_items['ratings'] = df_items['ratings'].apply(lambda x: round(x, 1))
        item_ids = set(df_items.product_id_int.tolist())

        item_features = df_items[['category', 'price', 'ratings']].apply(
                    lambda x: ','.join(x.map(str)), axis=1)
        item_features = item_features.str.split(',')
        item_features = item_features.apply(pd.Series).stack().reset_index(drop=True)
        
        id_item_features = df_items[['product_id_int', 'category', 'price', 'ratings']].apply(
            lambda row: (row['product_id_int'], [str(row['category']), str(row['ratings'])]), axis=1
        )
        
        df_interaction['is_purchased'] = df_interaction['purchase_date'].apply(lambda x : 1 if x is not None else None)
        df_interaction = df_interaction[
            ['customer_id_int', 'product_id_int', 'is_purchased']
        ].groupby(['customer_id_int', 'product_id_int']).sum().reset_index()
        df_interaction = pd.merge(df_interaction, df_items, on=['product_id_int'], how='left')
        df_interaction['weight'] = df_interaction['is_purchased']

        df_interaction = df_interaction[['customer_id_int', 'product_id_int', 'weight']]
        interactions = list(zip(df_interaction.customer_id_int, df_interaction.product_id_int, df_interaction.weight))

        self.user_features = user_features
        self.item_features = item_features
        self.interactions = interactions
        self.item_ids = item_ids
        self.id_item_features = id_item_features
        self.user_ids = user_ids
        self.id_user_features = id_user_features
        return

    def map_user_id_int(self, df):
        map_user_id = {}
        for ix, user in enumerate(df.customer_id.unique()):
            map_user_id[user] = ix
        self.map_user_id = map_user_id

    def map_item_id_int(self, df):
        map_item_id = {}
        for ix, item in enumerate(df.product_id.unique()):
            map_item_id[item] = ix
        self.map_item_id = map_item_id

    def fit_dataset(self):
        dataset = Dataset()
        dataset.fit(
            self.user_ids,
            self.item_ids,
            item_features=self.item_features,
            user_features=self.user_features)
        self.dataset = dataset

    def build_dataset(self):
        self.interactions, self.weight = self.dataset.build_interactions(self.interactions)
        self.item_features = self.dataset.build_item_features(self.id_item_features)
        self.user_features = self.dataset.build_user_features(self.id_user_features)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, user_id):
        purchased_ids = set(self.df_interaction.loc[self.df_interaction.customer_id == user_id].product_id_int.tolist())
        item_ids = list(set(self.df_items.loc[~self.df_items.product_id_int.isin(purchased_ids)].product_id_int.tolist()))
        df_last_buy = self.df_items.loc[self.df_items.product_id_int.isin(purchased_ids)]
        scores = self.model.predict(self.map_user_id[user_id], item_ids, self.item_features, self.user_features)
        df_recommend = self.df_items.loc[self.df_items.product_id_int.isin(item_ids)]
        df_recommend['score'] = scores
        df_recommend = df_recommend.sort_values('score', ascending=False).iloc[:10]
        return df_last_buy, df_recommend
