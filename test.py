import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
data_df=pd.read_csv("data_files/channel_data.csv",encoding='unicode_escape')

converted_products = data_df.groupby(['channel_name', 'product_name'])['conversion'].sum().reset_index()
max_converted_products = converted_products.loc[converted_products.groupby('channel_name')['conversion'].idxmax()]

print(max_converted_products)

