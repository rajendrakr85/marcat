from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
# since the configs folder is outside the directory the file is in, we need to add the parent path where the configs folder is present to be able to import the file from the configs folder
import sys
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import glob
import csv

sys.path.append(os.getcwd())

'''
  Content based algorithm gives the upsell recommendations.
'''


class ContentBased():
    '''
        This class is useful for predicting upsell recommendations.
        Update from the previous class:
            1. Here, since the updated preprocessed data is being used which has more than 20k rows,
            creating a matrix for similarity will be very RAM Heavy. Thus, instead of that, the similarity
            is generated between the vectors and the selected vector on the fly.
            2. One subsidiary can purchase the same material for more than one use, this version of code gives one recommendation
            for each one of them (if present).
        ------------------------------------
        Input Data:
        ----------------------
            Content based collaborative filtering will be working for Pharma solutions and H&B Businnes unit text.
            It consists of Pharma and H&B BU data.
        Output:
        -------------------------------------
        Top 2 recommendations
        Class Attributes:
        --------------------
        data=prepreocessed_data
        product_name="PRODUCT_NAME"
        recommendations="Recommendations"
        end_use_l1="CATEGORY_MARKET_L1"
        end_use_l2="SUB_CATEGORY_MARKET_L2"
        price="Price"
        Class Methods:
        ------------------------------
        PrepareDataset(data):
            Preprocessing the input dataset. It returns processed data set.
        CreatingVector():
            Creates the vectors for calculating the similarities for product attributes.
        get_content_based_recommendations():
            returns recommendations by using similarity scores
        get_recommendations(product):
            For the given target material based on the applied filters, it will recommend top 2 Upsell recommendations.
         '''

    def __init__(self, data):
        self.data = data
        self.product_name = 'PRODUCT_NAME'
        self.price = "PRICE"
        self.category = 'CATEGORY'

    def PrepareDataset(self):
        '''
            Preprocessing the input dataset
            -------------------------------
            Input:
            ----------------
                data: DataFrame
        '''
        self.data[self.product_name] = self.data[self.product_name].fillna('')
        self.data[self.product_name] = self.data[self.product_name].str.strip()
        # self.data[self.recommendations] = self.data[self.recommendations].fillna('')

    def create_indices_df_and_vectors(self):
        '''
            This function creates a dataframe containing the ownership, customer, product
            names and their corresponding indices where the combination exists in the self.data
            dataframe. It also creates the matrix containing the vectors for each of the
            recommendation value in the dataframe.
        '''
        # creating the indices dataframe
        # self.indices = self.data[[self.ownership_customer, self.customer_name, self.product_name]]
        self.indices = self.data[[self.category, self.price, self.product_name]]
        # getting the index
        self.indices['Index'] = self.indices.index
        # creating tfidvectorizer
        tfidfVectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                          analyzer='word', stop_words='english', ngram_range=(1, 3))
        self.vectorMatrix = tfidfVectorizer.fit_transform(self.data[self.product_name])

    def get_content_based_recommendations(self, index, title):
        '''
            This function gives the recommended products for the given index. The similarity score (SIG_SCORE) is calculated suing the sigmoid similarity function.
            Input:
            -----------
            index: One index from the list of the indices for the material, customer, ownership combination.
                    This has to be an integer and not an array/list/iterable.
            Output:
            --------------
            Recommendations : List
        '''
        # getting the vector
        vec = self.vectorMatrix[index]
        # calculating the similarities
        sims = sigmoid_kernel(self.vectorMatrix, vec)
        # enumerating the similarity list
        sig_scores = list(enumerate(sims.flatten()))
        # sorting in reverse
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[0:5000]
        # getting product indices from the sorted list
        product_indices = [i[0] for i in sig_scores]
        # returning recommendations based on the given target product name (title)
        recommendation = self.data[self.product_name].iloc[product_indices].values
        # rearranging them
        rearranged = self.rearrange_recommendations(recommendation, title)

        return rearranged

    def rearrange_recommendations(self, combined_list, item):
        '''
            It returns combined list of recommendations based on the given target product_name and there recommendations.
            Output: (List)
            ------------------
            first_list + second_list: recommendations of the given product name where given product name is not included.
        '''
        item_list = list(item.split(" "))
        item_to_comp = item_list[0]
        first_list = [x for x in combined_list if item_to_comp in x]
        second_list = [x for x in combined_list if
                       item not in x]  # recommendations where given product name is not included.
        return first_list + second_list

    # def get_recommendations(self, product, customer_name='', ownership_customer=''):
    def get_recommendations(self, product, price, category):
        '''
            It returns recommended products for the given target material based on the below filters.
                Filters: 1) Price
                         2) CATEGORY_MARKET_L1
                         3) SUB_CATEGORY_MARKET_L2
            Input:
            -------------------------
            product: Target product name (str)
            Output: (list)
            ------------------
            Top 2 recommendations for the given target product_name
        '''
        # create the vectors and the indices df
        self.create_indices_df_and_vectors()
        # getting the indices from the indices df for the input
        idx = self.indices[(self.indices[self.price] == price) &
            (self.indices[self.category] == category) &
            (self.indices[self.product_name] == product)]['Index'].values
        # looping over each of the indices and generating the results
        output_recs = []
        for i in idx:
            # getting the recommended materials
            rec_mats = self.get_content_based_recommendations(i, product)
            if len(rec_mats) == 0:
                cb_recommendation = ['No Recommendation']
                output_recs.append(cb_recommendation)
            else:
                # getting all info about the recommendations
                Recommendations = self.data[self.data[self.product_name].isin(rec_mats)]
                # applying the price filter for the target material
                Recommendations = Recommendations.sort_values(self.product_name, ascending=True)
                rec_product = Recommendations[
                    (Recommendations[self.price] == price) &
                    (Recommendations[self.category] == category) &
                    (Recommendations[self.product_name] == product)].loc[i]
                prod_price = rec_product[self.price]
                Recommendations = Recommendations[Recommendations[self.price] > prod_price]
                # Removing any rows with the same material as the product itself
                Recommendations = Recommendations[Recommendations[self.product_name] != product]
                Recommendations = Recommendations[self.product_name].unique()[
                                  :5]
                output_recs.extend(Recommendations.tolist())
        output_recs = list(set(output_recs))
        return output_recs


