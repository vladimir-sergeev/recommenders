import pandas as pd
import scipy.sparse
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import TruncatedSVD

class ItemBasedRecommender():
    
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.user_item = self.get_user_item()
        self.calculate_distances_function = self.distances_to_mean_vector
        
    def get_user_item(self):
        '''Create user-item matrix from train_data.'''
        users = [user for user, items in self.train_data.items() for _ in range(len(items))]
        items = [item for items in self.train_data.values() for item in items]
        user_item = scipy.sparse.csc_matrix((np.ones(len(users)), (users, items)))
        return user_item
    
    def fit(self, components_number, save_path=False):
        '''
        Get singular value decomposition of user-item matrix.
        
        components_number -- value for reduced dimension
        save_path -- path to file to save svd-model (if False do not save)
        '''
        svd = TruncatedSVD(n_components=components_number)
        self.user_vector = svd.fit_transform(self.user_item)
        self.item_vector = svd.components_
        self.explained_variance = svd.explained_variance_ratio_.sum()
        if save_path:
            with open(save_path, 'wb') as out:
                pickle.dump(svd, out)
                
    def set_svd(self, path_to_svd):
        '''
        Load svd-model from file.
        
        path_to_svd -- path to file with svd-model
        '''
        with open(path_to_svd, 'rb') as inp:
            svd = pickle.load(inp)
        self.user_vector = svd.transform(self.user_item)
        self.item_vector = svd.components_
        self.explained_variance = svd.explained_variance_ratio_.sum() 
    
    def mean_distances_to_vectors(self, items_list):
        '''Calculate mean distances between each item from item_vector and items form list.'''
        return np.mean(cosine_distances(self.item_vector[:, items_list].T, self.item_vector.T), 0)
        
    def distances_to_mean_vector(self, items_list):
        '''
        Find mean-vector of items from list. Calculate distances between mean-vector and each item form item_vector.
        '''
        item_mean = np.mean(self.item_vector[:, items_list], 1)
        return cosine_distances([item_mean], self.item_vector.T)[0]
        
    def set_distances_calculation_function(self, calc_function):
        self.calculate_distances_function = calc_function
    
    def predict_on_items(self, items_list, top_k):
        '''
        Calculate distances between each item and items from list. Get top predictions based on list of items.
        '''
        distance = self.calculate_distances_function(items_list)
        recommendation = np.argsort(distance)
        return recommendation[:top_k]
    
    def predict_for_user(self, user_id, top_k):
        '''Get top predictions for user.'''
        return self.predict_on_items(self.train_data[user_id], top_k)
    
    def predict_for_top_n_users(self, top_k, user_number=100, path_to_save=False):
        '''
        Get pridictions for several top users.
        
        path_to_save -- path to file to save results (if False do not save)
        '''
        recommendations = {key: [] for key in range(1, user_number + 1)}
        for user_id in range(1, user_number + 1):
            recommendations[user_id] = self.predict_for_user(user_id, top_k)
        if path_to_save:
            with open(path_to_save, 'wb') as out:
                pickle.dump(recommendations, out)            
        return recommendations
    
    def get_mapk(self, predicted, k=100):
        '''Get mean average precision at k in train, validation and test data.'''
        train_mapk = mapk(list(self.train_data.values()), predicted.values(), k)
        validation_mapk = mapk(list(self.validation_data.values()), predicted.values(), k)
        test_mapk = mapk(list(self.test_data.values()), predicted.values(), k)
        return train_mapk, validation_mapk, test_mapk

    
def load_data():
    with open('train_indices.pickle', 'rb') as input:
        train_indices = pickle.load(input)
    with open('test_indices.pickle', 'rb') as input:
        test_indices = pickle.load(input)
    with open('validation_indices.pickle', 'rb') as input:
        validation_indices = pickle.load(input)
        
    orders = pd.read_csv('orders.csv')[['user_id', 'order_id']]
    train_ids = orders.loc[train_indices]['order_id']
    test_ids = orders.loc[test_indices]['order_id']    
    validation_ids = orders.loc[validation_indices]['order_id']
    
    order_products_prior = pd.read_csv('order_products__prior.csv')[['product_id', 'order_id']]
    order_products_train = pd.read_csv('order_products__train.csv')[['product_id', 'order_id']]
    order_products = pd.concat([order_products_prior, order_products_train])
    orders = pd.merge(order_products, orders, on = 'order_id')
    
    orders_tmp = orders[orders['order_id'].isin(train_ids)].drop('order_id', 1).drop_duplicates()
    train_data = dict(orders_tmp.groupby('user_id')['product_id'].apply(list))
    
    orders_tmp = orders[orders['order_id'].isin(validation_ids)].drop('order_id', 1).drop_duplicates()
    validation_data = dict(orders_tmp.groupby('user_id')['product_id'].apply(list))
    
    orders_tmp = orders[orders['order_id'].isin(test_ids)].drop('order_id', 1).drop_duplicates()
    test_data = dict(orders_tmp.groupby('user_id')['product_id'].apply(list))
    
    return train_data, test_data, validation_data

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
