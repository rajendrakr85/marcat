import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import glob
import csv

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

csv_files = glob.glob('Data_files/rec-data/product.csv')
all_data = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_data = pd.concat([df, all_data])

all_data = all_data.sort_index()
# all_data = all_data.replace('...', '').replace(',', '').replace('|', '')
all_data['name'] = all_data['name'].fillna('')
all_data = all_data.dropna()
all_data = all_data.drop_duplicates()
all_data = all_data.sort_values('name')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(all_data['name'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend_products(product_name, cosine_sim=cosine_sim, df=all_data):
    idx = df[df['name'] == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [(i, score) for i, score in sim_scores if i != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[product_indices]


# product_word_list = all_data['name'].str.split(' ').tolist()
#
#
# def create_bow(prddataset):
#     bow = {}
#     for prd in prddataset:
#         bow[prd] = 1
#     return bow
#
#
# def recommend_products_bags_of_words(product_name, cosine_sim=cosine_sim, df=all_data):
#     bags_of_words = [create_bow(prds) for prds in product_word_list]


recommended_products = recommend_products('BSB HOME Premium Cotton Elastic Fitted Bedsheets with 2 King Size Pillow Covers | Double Bed with All Around Elastic 180 T')
# recommended_products = recommend_products('Kozdiko Black Shark Fin Signal Antenna for - Maruti Suzuki Swift Dzire Model')
print(recommended_products)
