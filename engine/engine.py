import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM
from sklearn.model_selection import train_test_split
from lightfm.evaluation import precision_at_k
import pickle


##############################################################################
# 0. importing dataset
data = pd.read_csv('Online Retail.csv')
# ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'],


##############################################################################
# 1. cleaning data

# data.info() shows CustomerID and Description columns contains some nan values. we need both of this coloumns in this analysis so we drop rows with nan values.
data.dropna(how='any', inplace=True)
data.reset_index(drop=True, inplace=True)

# removing invalid rows which has chars in their StockCode
CodeTypes = list(map(lambda codes: any(char.isdigit() for char in codes), data['StockCode']))
invalidCodes = [i for i,v in enumerate(CodeTypes) if v == False]
data.drop(invalidCodes, inplace=True)
data.reset_index(drop=True, inplace=True)


data.drop(data[(data['Quantity']<=0)].index, inplace = True)
data.reset_index(drop=True, inplace=True)

data.drop(data[data['UnitPrice'] <= 0].index, inplace = True)
data.reset_index(drop=True, inplace=True)

data.drop(data[data['InvoiceNo'].str.contains('C') == True].index, inplace = True)
data.reset_index(drop=True, inplace=True)

data.drop(data[data['Description'].str.contains('?',regex=False) == True].index, inplace = True)
data.reset_index(drop=True, inplace=True)

for idx,val in data[data['Description'].isna()==True]['StockCode'].items():
    if pd.notna(data[data['StockCode'] == val]['Description']).sum() != 0:
        data['Description'][idx] = data[data['StockCode'] == val]['Description'].mode()[0]
    else:
        data.drop(index = idx, inplace = True)

data['Description'] = data['Description'].astype(str)
data.reset_index(drop=True, inplace=True)




##############################################################################
# 2. mapping
users = data['CustomerID'].unique()
products = data['Description'].unique()

user_to_idx = {}
idx_to_user = {}
for u_index, u_id in enumerate(users):
    user_to_idx[u_id] = u_index
    idx_to_user[u_index] = u_id

product_to_idx = {}
idx_to_product = {}
for p_index, p_id in enumerate(products):
    product_to_idx[p_id] = p_index
    idx_to_product[p_index] = p_id




##############################################################################
# 3. user product interaction
user_product = data[['CustomerID','Description','Quantity']]
user_product['count'] = user_to_product['Quantity']

train, _ = train_test_split(user_product, test_size=0.2)

user_product_train = train.groupby(['CustomerID','Description'], as_index = False)['count'].sum()
user_product_test = user_product.groupby(['CustomerID','Description'], as_index = False)['count'].sum()
#in user_to_product_test we groupby user_to_product because in test set we want to store actual rates because we are dealing with rates! and not counts!

def get_interaction_coo_matrix(df, scale=1.0):
    row = df['CustomerID'].apply(lambda x: user_to_idx[x]).values
    col = df['Description'].apply(lambda x: product_to_idx[x]).values
    value = (df['count'].values)*(scale)
    return coo_matrix((value, (row, col)), shape = (len(user_to_idx), len(product_to_idx)))

userXproduct_train = get_interaction_coo_matrix(user_product_train)
userXproduct_test = get_interaction_coo_matrix(user_product_test, 0.8) # we scale test set rates for balancing the range of ratings in the train set and test set. because train set size is 80% of the test set size



##############################################################################
# 4. training
model = LightFM(loss = "warp")
model.fit(userXproduct_train,
          epochs=50, 
          num_threads=4)



##############################################################################
# 5. evaluating
precision_at_k(model = model, test_interactions = userXproduct_train, k=10).mean()
precision_at_k(model = model, test_interactions = userXproduct_test, k=10).mean()



##############################################################################
# 6. predicting
def recom_to_sample(model, user):
    userindex = user_to_idx.get(user, None)
    if userindex == None:
        return None
    users = [userindex]
#    known_positives = products[userXproduct.tocsr()[userindex].indices]
    scores = model.predict(user_ids = users, item_ids = np.arange(userXproduct.shape[1]))
    top_items = products[np.argsort(-scores)]
    return top_items[:10]




##############################################################################
# 7. save model
# alternatively saving LUT of answers
LUT={}
for i in users:
    LUT[i]=recom_to_sample(model,i)

with open('saved_lut.pkl','wb') as f:
    pickle.dump(LUT, f)

##############################################################################
# 8. example 
recom_to_sample(model,12346)