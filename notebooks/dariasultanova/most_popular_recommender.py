import pandas as pd
from recommender import Recommender


class MostPopularRecommender(Recommender):

    def __init__(self, train_df, valid_df, test_df, orders_df, order_products_df):
        self.train = train_df
        self.valid = valid_df
        self.test = test_df
        self.orders_df = orders_df
        self.order_products_df = order_products_df

    def fit(self):
        merged = pd.merge(self.train, self.order_products_df, on='order_id')[['order_id', 'product_id']]
        self.product_ids = merged['product_id'].value_counts().index.values.tolist()

    def predict(self, user_ids, top_k):
        recommendations = pd.DataFrame()
        recommendations['user_id'] = [i for i in user_ids]
        recommendations['predictions'] = [self.product_ids[:top_k] for i in user_ids]
        return recommendations.sort_values(by=['user_id'])
