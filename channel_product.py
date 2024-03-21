import dash
from dash import dcc, html
import flask
from dash import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import content_based
import plotly.graph_objs as go
import base64

cdf = pd.read_csv('data_files/channel_data.csv')
cdf.drop_duplicates(inplace=True, keep="last")
converted_products = cdf.groupby(['category', 'price', 'channel_name', 'product_name'])[
    'conversion'].sum().reset_index()
max_converted_products = converted_products.loc[converted_products.groupby(['channel_name'])['conversion'].idxmax()]
max_converted_products.index = range(len(max_converted_products))

app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    # Sidebar menu
    html.Aside(
        children=[
            html.Img(src='images/img.png', alt='Company Logo',
                     style={'width': '80px', 'height': '80px', 'margin-bottom': '20px'}),
            html.H2('Marketing Menu', style={'margin-bottom': '20px', 'padding-bottom': '30px'}),
            html.Ul([
                html.Li(html.A('Home', href='/', style={'text-decoration': 'none', 'color': '#333'})),
                html.Li(html.A('About', href='/about', style={'text-decoration': 'none', 'color': '#333'})),
                html.Li(html.A('Contact', href='/contact', style={'text-decoration': 'none', 'color': '#333'}))
            ], style={'list-style-type': 'none', 'padding': 0, 'margin': 0})
        ],
        style={'background-color': '#f0f0f0', 'padding': '20px', 'width': '200px', 'height': '100vh',
               'position': 'fixed', 'left': 0, 'top': 0}
    ),
    # Main content area
    html.Div([
        html.Div([
            # Page title
            html.H1('MarCat GRE Solution', style={'textAlign': 'center', 'margin-top': '20px'}),
            # Charts section
            html.Div([
                # Pie chart
                dcc.Graph(
                    id='pie-chart',
                    figure={
                        'data': [
                            go.Pie(
                                labels=cdf['channel_name'],
                                values=cdf['conversion'],
                                hole=0.4,
                                textinfo='percent+label',
                                hoverinfo='label+percent',
                                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                            )
                        ],
                        'layout': {
                            'title': 'Pie Chart on Conversion',
                            'margin': dict(t=50, b=0, l=0, r=0),
                            'legend': dict(orientation='h', x=0.2, y=-0.1),
                            'height': 300
                        }
                    }
                ),
                # Bar chart
                dcc.Graph(
                    id='bar-chart',
                    figure={
                        'data': [
                            go.Bar(
                                x=cdf['channel_name'],
                                y=cdf['conversion'],
                                marker=dict(color='#1f77b4')
                            )
                        ],
                        'layout': {
                            'title': 'Bar Chart on Conversion',
                            'margin': dict(t=50, b=50, l=50, r=50),
                            'height': 300
                        }
                    }
                )
            ], style={'display': 'flex', 'justify-content': 'space-around', 'flex-wrap': 'wrap', 'margin-top': '20px'}),

            html.Div([
                html.H1('Top Product on Channels'),
                html.Div([
                    html.A(html.Button('Download final channel data and recommended products on channels'),
                           id='btn-xlsx-download', href='/download-file')
                ], style={'display': 'block', 'margin-bottom': '20px', 'align': 'right', 'color': '#fff',
                          'border': '10px', 'padding': '10px 20px',
                          'cursor': 'pointer', 'border-radius': '20px', 'font-size': '16px'}),
                dash_table.DataTable(
                    id='table1',
                    columns=[{'name': 'Category', 'id': 'category'}, {'name': 'Price', 'id': 'price'},
                             {'name': 'Channel Name', 'id': 'channel_name'}, {'name': 'Product', 'id': 'product_name'},
                             {'name': 'Conversion', 'id': 'conversion'}],
                    data=max_converted_products.to_dict('records'),
                    style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'fontSize': '12px', 'padding': '5px', 'minWidth': '10px',
                                'maxWidth': '300px'},
                    style_table={'width': '100%', 'maxHeight': '300px', 'overflowY': 'scroll',
                                 'border': '1px solid #ddd'},
                    fixed_rows={'headers': True, 'data': 0}
                ),
                html.H1('Recommended Products on Channel')
            ]),
            html.Div(id='table2-container')
        ], style={'margin-left': '220px', 'padding': '20px'}),

    ])
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
    content_processed = []
    if channel_product['category'] == 'Home Decor':
        content_processed = pd.read_csv('Data_files/Home_Dcor.csv')
    elif channel_product['category'] == 'Air Conditioner':
        content_processed = pd.read_csv('data_files/Air Conditioner.csv')
    elif channel_product['category'] == 'Car Accessories':
        content_processed = pd.read_csv('data_files/Car Accessories.csv')
    content_recoms = content_based.ContentBased(content_processed)

    recommended_products = content_recoms.get_recommendations(channel_product['product_name'], channel_product['price'],
                                                              channel_product['category'])
    subtable = []
    if len(recommended_products) > 0:
        table_rows = [{'Row Number': i + 1, 'Recommended_Products': product} for i, product in
                      enumerate(recommended_products)]

        with pd.ExcelWriter('results/final_data.xlsx') as writer:
            max_converted_products.to_excel(writer, sheet_name='Final products on each chanel', index=False)
            pd.DataFrame(recommended_products).to_excel(writer, sheet_name='Recommended Products', index=False)

        subtable = dash_table.DataTable(
            id='subtable',
            columns=[{'name': 'Row Number', 'id': 'Row Number'},
                     {'name': 'Recommended Products', 'id': 'Recommended_Products'}],
            data=table_rows,
            style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'fontSize': '12px', 'padding': '10px', 'height': 'auto'},
            style_table={'headers': 'True', 'width': '60%', 'margin': 'auto', 'marginTop': '20px',
                         'border': '1px solid #ddd', 'overflowX': 'auto'}
        )

        return subtable
    else:
        return html.H1('No Product Recommended on ' + channel_product['channel_name'],
                       style={'textAlign': 'center', 'fontSize': '32px', 'color': 'red'})


@app.server.route("/download-file")
def download_file():
    # Path to the file to be downloaded
    file_path = "results/final_data.xlsx"
    return flask.send_file(file_path, as_attachment=True)


# Run the app
if __name__ == '__main__':
    app.run_server(port=8080, debug=True)
