# Aficionado_WWT_Comp2025

Requirements:
pandas
numpy
gensim
xgboost
scikit-learn
openpyxl

This repository contains the codebase for a recommendation system designed to predict the next item a customer is likely to add to their shopping cart. The solution was developed for a data science competition and uses a two-stage machine learning approach involving item embeddings and a Learning-to-Rank model.

Project Overview

The primary goal is to provide real-time product recommendations based on the current contents of a user's shopping basket. Given a set of items, the model predicts the top 3 most probable items to be added next.

Technical Approach

The recommendation engine is built using a hybrid approach that combines Natural Language Processing techniques with a powerful gradient-boosted ranking model.

1. Item Embeddings with Word2Vec

The foundation of this model is understanding the relationships between items. We treat each customer's order as a "sentence" and the items within it as "words".

    A Word2Vec (Skip-Gram) model is trained on the historical order data.

    This process learns a 100-dimensional vector representation (embedding) for each item, capturing contextual relationships. Items frequently purchased together will have similar vectors.

2. Candidate Generation

To make predictions efficiently, we first generate a small, relevant list of candidate items for each input basket, rather than scoring every item in the inventory. Candidates are generated from two sources:

    Similarity-based: Items that are most similar to the ones already in the basket, based on the cosine similarity of their Word2Vec embeddings.

    Popularity-based: Globally popular items are also included to account for common purchases.

3. Feature Engineering

For the ranking model to make an informed decision, we engineer a rich set of features for each (input_basket, candidate_item) pair. Key features include:

    Basket Cohesion: Average cosine similarity between the candidate item's vector and the vectors of items already in the basket.

    Item Popularity: Global, city-level, and state-level frequency of the candidate item.

    Contextual Features: One-hot encoded features for the order channel (e.g., Pickup, Delivery) and customer type (e.g., guest, registered).

    Basket Size: The number of items currently in the cart.

4. Learning-to-Rank with XGBRanker

The core of the prediction is an XGBRanker model, a gradient boosting algorithm specifically designed for ranking tasks.

    The model is trained on historical orders where we artificially "mask" an item from a completed basket and task the model with ranking it highly among other candidates.

    GridSearchCV is used to systematically tune the model's hyperparameters, optimizing for a custom Recall@3 metric. This ensures the model is fine-tuned to place the correct item within the top 3 recommendations.

Link to Google colab:
https://colab.research.google.com/drive/1NEGp_6FeQrZe_QIeORTi1EafMfYXl6TE?usp=sharing
