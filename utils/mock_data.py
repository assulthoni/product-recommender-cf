import os
import pandas as pd
import numpy as np

N_CUSTOMER = 20
N_PRODUCT = 500
SEED = 24

PATH = os.path.dirname(os.path.abspath(__file__))
CUSTOMER_IDS = [i for i in range(1, N_CUSTOMER + 1)]
PRODUCT_IDS = [i for i in range(101, N_PRODUCT + 101)]
np.random.seed(SEED)


def generate_product_details():
    df_original = pd.read_csv(PATH.replace('utils', 'dataset') + '/product_details.csv', sep=';')
    df_original = df_original.loc[:, ~df_original.columns.str.contains('^Unnamed')]

    df_new = pd.DataFrame()
    df_new['product_id'] = [p for p in PRODUCT_IDS if p not in df_original.product_id.unique()]
    df_new['category'] = [
        np.random.choice(df_original.category.unique()) for _ in range(N_PRODUCT - df_original.product_id.count())
    ]
    df_new['price'] = [
        round(x) for x in np.random.choice(np.arange(20, 1500, 10), size=(N_PRODUCT - df_original.product_id.count()))
    ]
    df_new['ratings'] = [
        round(x, 1) for x in np.random.uniform(low=2.0, high=5.0, size=(N_PRODUCT - df_original.product_id.count()))
    ]
    df_new = pd.concat([df_original, df_new])
    df_new = df_new.reset_index(drop=True)
    return df_new


def generate_customer_interections():
    df_original = pd.read_csv(PATH.replace('utils', 'dataset') + '/customer_interactions.csv', sep=',')
    df_original = df_original.loc[:, ~df_original.columns.str.contains('^Unnamed')]
    
    df_new = pd.DataFrame()
    df_new['customer_id'] = [c for c in CUSTOMER_IDS if c not in df_original.customer_id.unique()]
    df_new['page_views'] = [
        round(x) for x in np.random.uniform(low=5, high=50, size=(N_CUSTOMER - df_original.customer_id.count()))
    ]
    df_new['time_spent'] = df_new['page_views'] * np.random.randint(5, 10)
    
    df_new = pd.concat([df_original, df_new])
    df_new = df_new.reset_index(drop=True)
    return df_new


def generate_purchase_history():
    size = 100
    # three months historical data
    datelist = pd.date_range(start="2023-01-01", end="2023-03-31").tolist()

    df_original = pd.read_csv(PATH.replace('utils', 'dataset') + '/purchase_history.csv', sep=';')
    df_original = df_original.loc[:, ~df_original.columns.str.contains('^Unnamed')]
    
    df_new = pd.DataFrame()
    df_new['customer_id'] = np.random.choice(CUSTOMER_IDS, size=(size - df_original.product_id.count()))
    df_new['product_id'] = np.random.choice(PRODUCT_IDS, size=(size - df_original.product_id.count()))
    df_new['purchase_date'] = np.random.choice(datelist, size=(size - df_original.product_id.count()))
    
    df_new = pd.concat([df_original, df_new])
    df_new = df_new.reset_index(drop=True)
    return df_new


def save_mock_data(df, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + filename, index=False, mode='w+')


if __name__ == '__main__':
    df_product = generate_product_details()
    save_mock_data(df_product, './dataset/new', 'product_details.csv')
    df_customer_int = generate_customer_interections()
    save_mock_data(df_customer_int, './dataset/new', 'customer_interactions.csv')
    df_purchase = generate_purchase_history()
    save_mock_data(df_purchase, './dataset/new', 'purchase_history.csv')
