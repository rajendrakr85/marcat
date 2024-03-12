import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import content_based

cdf = pd.read_csv('Data_files/channel_data.csv')
converted_products = cdf.groupby(['channel_name', 'product_name', 'price', 'category'])[
    'conversion'].sum().reset_index()

max_converted_products = converted_products.loc[converted_products.groupby('channel_name')['conversion'].idxmax()]
max_converted_products.index = range(len(max_converted_products))

all_data = pd.read_csv('Data_files/product.csv')
all_data['PRODUCT_NAME'] = all_data['PRODUCT_NAME'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(all_data['PRODUCT_NAME'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Top Product on Channels'),
    dash_table.DataTable(
        id='table1',
        columns=[{'name': col, 'id': col} for col in max_converted_products.columns],
        data=max_converted_products.to_dict('records'),
        style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'center', 'fontSize': '12px', 'padding': '10px'},
        style_table={'width': '100%', 'maxHeight': '300px', 'overflowY': 'scroll', 'border': '1px solid #ddd'},
        fixed_rows={'headers': True, 'data': 0}
    ),
    html.H1('Recommended Products on Channel'),
    html.Div(id='table2-container')
])


@app.callback(
    Output('table2-container', 'children'),
    [Input('table1', 'active_cell')]
)
def update_recommendations(active_cell):
    if not active_cell:
        return []

    row_index = active_cell['row']
    channel_product = max_converted_products.loc[row_index]
    # print(channel_product)
    content_processed = pd.read_csv('Data_files/Home_Dcor.csv')
    content_recoms = content_based.ContentBased(content_processed)
    recommended_products = content_recoms.get_recommendations(channel_product['product_name'], channel_product['price'],
                                                              channel_product['category'])
    for prd in recommended_products:
        print(prd)
    subtable = dash_table.DataTable(
        id='subtable',
        columns=[{'name': 'Recommended Products', 'id': 'Recommended_Products'}],
        # columns=[{'name': col, 'id': col} for col in max_converted_products.columns],
        #data=recommended_products,
        data=[{'Recommended_Products': ', '.join(recommended_products)}],
        style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'fontSize': '12px', 'padding': '10px', 'height': 'auto'},
        style_table={'headers': 'True', 'width': '50%', 'margin': 'auto', 'marginTop': '20px',
                     'border': '1px solid #ddd', 'overflowX': 'auto'}
    ),
    html.Div(id='output')

    return subtable


def iterate_table_data(rows):
    text = ''
    for row in rows:
        for key, value in row.items():
            text += f'{key}: {value}, '
        text += '<br>'
    return html.Div([html.P(text)])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
