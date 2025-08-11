!pip install pandas numpy gensim xgboost openpyxl --quiet

import pandas as pd
import numpy as np
import xgboost as xgb
import random
import json
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBRanker

# Load Data
# Ensure these file paths are correct in your Google Drive
orders = pd.read_csv('order_data.csv', usecols=['CUSTOMER_ID', 'STORE_NUMBER', 'ORDERS', 'ORDER_CHANNEL_NAME'])
customers = pd.read_csv('customer_data.csv')
stores = pd.read_csv('store_data.csv')
test_df = pd.read_csv('test_data_question.csv')

def extract_item_names(order_str):
    try:
        # Replace single quotes with double quotes for valid JSON parsing
        parsed = json.loads(order_str.replace("'", '"'))
        items = []
        for order in parsed.get("orders", []):
            for item in order.get("item_details", []):
                name = item.get("item_name", "").strip()
                # Filter out empty or irrelevant item names
                if name and name.lower() not in {"order memo not paid"}:
                    items.append(name)
        return items
    except Exception:
        return []

orders['ITEMS'] = orders['ORDERS'].apply(extract_item_names)
# Keep only orders with at least 2 items to form meaningful pairs
orders = orders[orders['ITEMS'].apply(lambda x: len(x) >= 2)]

# Merge customer/store data to enrich the order information
orders = orders.merge(customers, on='CUSTOMER_ID', how='left')
orders = orders.merge(stores, on='STORE_NUMBER', how='left')
test_df = test_df.merge(customers, on='CUSTOMER_ID', how='left')
test_df = test_df.merge(stores, on='STORE_NUMBER', how='left')

# Train Word2Vec model on the item sequences
class Corpus:
    def __init__(self, df):
        self.df = df
    def __iter__(self):
        yield from self.df['ITEMS']

w2v = Word2Vec(Corpus(orders), vector_size=100, window=10, min_count=1, sg=1, workers=2, epochs=10)
item_freq = orders.explode('ITEMS')['ITEMS'].value_counts(normalize=True).to_dict()

# Pre-calculate popular items by city and state for feature engineering
top_city_items_df = orders.explode('ITEMS').groupby(['CITY', 'ITEMS']).size().reset_index(name='count')
top_city_items = top_city_items_df.groupby('CITY')['ITEMS'].apply(set).to_dict()
top_state_items_df = orders.explode('ITEMS').groupby(['STATE', 'ITEMS']).size().reset_index(name='count')
top_state_items = top_state_items_df.groupby('STATE')['ITEMS'].apply(set).to_dict()

def cosine_sim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_features(input_items, candidate, channel, cust_type, city, state):
    vecs = [w2v.wv[i] for i in input_items if i in w2v.wv]
    cand_vec = w2v.wv[candidate] if candidate in w2v.wv else None
    avg_vec = np.mean(vecs, axis=0) if vecs else None
    cooccur_score = sum([1 for i in input_items if i != candidate and i in w2v.wv and candidate in w2v.wv])
    return {
        'basket_size': len(input_items),
        'item_freq': item_freq.get(candidate, 0),
        'avg_sim': cosine_sim(avg_vec, cand_vec),
        'channel_pickup': 1 if channel == 'Pickup' else 0,
        'channel_delivery': 1 if channel == 'Delivery' else 0,
        'cust_guest': 1 if cust_type == 'guest' else 0,
        'cust_registered': 1 if cust_type == 'registered' else 0,
        'cust_special': 1 if cust_type == 'special membership' else 0,
        'is_city_popular': 1 if city and (candidate in top_city_items.get(city, set())) else 0,
        'is_state_popular': 1 if state and (candidate in top_state_items.get(state, set())) else 0,
        'cooccur_score': cooccur_score
    }

def is_valid_item(item):
    return isinstance(item, str) and len(item) > 2 and ':' not in item and 'item_' not in item.lower()

# Build Training and Validation Data
train_orders, val_orders = train_test_split(orders, test_size=0.1, random_state=42)
X, y, group = [], [], []
val_data_for_eval = []

for _, row in val_orders.iterrows():
    cart = row['ITEMS']
    if len(cart) < 3:
        continue
    masked = random.choice(cart)
    inputs = [i for i in cart if i != masked]

    # Generate candidate items
    candidates = defaultdict(float)
    for item in inputs:
        if item in w2v.wv:
            for sim_item, score in w2v.wv.most_similar(item, topn=20):
                if sim_item not in inputs:
                    candidates[sim_item] += score
    for item, freq in item_freq.items():
        if item not in inputs:
            candidates[item] += 0.1 * freq

    cand_list = [c for c in candidates if is_valid_item(c)][:25]
    feats = [get_features(inputs, c, row['ORDER_CHANNEL_NAME'], str(row.get('CUSTOMER_TYPE', '')).lower(), row['CITY'], row['STATE']) for c in cand_list]
    labels = [1 if c == masked else 0 for c in cand_list]

    # Only include queries where the true item is in the candidate list
    if 1 not in labels:
        continue
    X.extend(feats)
    y.extend(labels)
    group.append(len(labels))
    val_data_for_eval.append((feats, labels, cand_list, masked))

X_df = pd.DataFrame(X)

def recall_at_3_scorer(estimator, X_matrix, y_true):
    recall_hits, total_queries = 0, 0
    for feats, labels, cand_list, masked in val_data_for_eval:
        X_block = pd.DataFrame(feats)
        preds = estimator.predict(X_block)
        top3 = [cand_list[i] for i in np.argsort(preds)[::-1][:3]]
        if masked in top3:
            recall_hits += 1
        total_queries += 1
    return recall_hits / total_queries if total_queries > 0 else 0

scorer = make_scorer(recall_at_3_scorer, greater_is_better=True)

# Hyperparameter grid for XGBRanker
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'n_estimators': [100, 200]
}

# Use GridSearchCV to find the best hyperparameters
# Note: cv is set to use the full validation set for training and testing in each "fold",
# which is appropriate for a custom scorer on a fixed validation set.
grid_search = GridSearchCV(xgb.XGBRanker(objective='rank:pairwise', eval_metric='ndcg@3', random_state=42),
                           param_grid, scoring=scorer, cv=[(np.arange(len(y)), np.arange(len(y)))], verbose=2, n_jobs=-1)
grid_search.fit(X_df, y, group=group)

print("Best Params:", grid_search.best_params_)
model = grid_search.best_estimator_

# Evaluate the best model on the validation set
final_recall = recall_at_3_scorer(model, None, None)
print(f"📊 Final Local Recall@3 on Validation Set: {final_recall:.4f}")

# Ensure all required item columns exist in test_df before predictions
required_items = ['item1', 'item2', 'item3', 'item4']
for col in required_items:
    if col not in test_df.columns:
        test_df[col] = ""

# Generate recommendations for the test set
recommendations = []
for _, row in test_df.iterrows():
    inputs = [row.get(f'item{i}', '') for i in range(1, 5) if pd.notna(row.get(f'item{i}', ''))]

    # Generate candidates for the test query
    candidates = defaultdict(float)
    for item in inputs:
        if item in w2v.wv:
            for sim_item, score in w2v.wv.most_similar(item, topn=20):
                if sim_item not in inputs:
                    candidates[sim_item] += score
    for item, freq in item_freq.items():
        if item not in inputs:
            candidates[item] += 0.1 * freq

    cand_list = [c for c in candidates if is_valid_item(c)][:25]
    feats = [get_features(inputs, c, row['ORDER_CHANNEL_NAME'], str(row.get('CUSTOMER_TYPE', '')).lower(), row['CITY'], row['STATE']) for c in cand_list]

    if not feats:
        recommendations.append(["", "", ""]) # Handle cases with no valid candidates
        continue

    # Predict scores and get top 3 recommendations
    preds = model.predict(pd.DataFrame(feats))
    top3 = [cand_list[i] for i in np.argsort(preds)[::-1][:3]]
    # Ensure we always have 3 recommendations, padding with empty strings if necessary
    top3.extend([""] * (3 - len(top3)))
    recommendations.append(top3)

# Format the final output file
output_cols = ['CUSTOMER_ID', 'ORDER_ID', 'item1', 'item2', 'item3', 'item4',
               'RECOMMENDATION 1', 'RECOMMENDATION 2', 'RECOMMENDATION 3']
final_output = test_df.copy()
final_output[['RECOMMENDATION 1', 'RECOMMENDATION 2', 'RECOMMENDATION 3']] = pd.DataFrame(recommendations, index=final_output.index)
final_output = final_output[output_cols]
final_output.to_csv('Aficionado_Recommendation.csv', index=False)

print("\nSubmission file saved to 'Aficionado_Recommendation.csv'")
