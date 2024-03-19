import base64
import dash
from dash import html,dcc
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from dash import dash_table
import flask
from datetime import date
from channel_attribution_models import channel_attributions
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")
import configparser
## Intial Configurations setup
config = configparser.ConfigParser()
config.read('configurations.ini')
total_customers = int(config.get('constants', 'total_customers'))


chnl_att = channel_attributions()
## Data Reading and cleaning
data_df=pd.read_csv("data_files/Data_cookie.csv",encoding='unicode_escape')
unique_channels = data_df['channel_name'].unique() ## Unique channels
data_df = data_df.sort_values(['is_converted','channel_name'],  ascending=[False, True])
data_df.drop_duplicates(inplace=True,keep="last")

# Data Preprocessin
data_processed = chnl_att.data_preprocessing(data_df)
print(data_processed)
num_conv=len(data_processed[data_processed["is_converted"]==1])
num_obs =len(data_processed)
# print(num_obs)

### Number of interaction before conversion

num_intaraction_df =data_processed.groupby("Paths_len").sum().reset_index()
# print(num_intaraction_df)

# ## Marko Chain model
marko_mtrx = chnl_att.marko_chain(data_processed)
for col in marko_mtrx.columns:
    marko_mtrx[col]=[round(val,2) for val in marko_mtrx[col]]


heatmap_marko = px.imshow(marko_mtrx,text_auto=True,aspect='auto',color_continuous_scale=px.colors.sequential.Blues)
heatmap_marko.update_layout(yaxis_title="Channels")

marko_op = chnl_att.removal_effects(marko_mtrx,num_conv/num_obs)
marko_op = pd.DataFrame(pd.Series(marko_op))
marko_op = marko_op.reset_index()
marko_op.columns = ["Channel", "Values"]


# First Touch Attribution
df_frst = pd.DataFrame()
df_frst['Channel'] = data_processed['First Touch']
df_frst['Conversion'] = data_processed["is_converted"]
df_frst = df_frst.groupby(['Channel']).sum().reset_index()
df_frst['Values'] = df_frst["Conversion"]/num_conv


# Last Touch Attribution
df_lst = pd.DataFrame()
df_lst['Channel'] = data_processed['Last Touch']
df_lst['Conversion'] = data_processed["is_converted"]
df_lst = df_lst.groupby(['Channel']).sum().reset_index()
df_lst['Values'] = df_lst["Conversion"]/num_conv


## Linear Touch With conversion filter
channel = []
conversion = []
temp_df = data_processed[data_processed["is_converted"]==1]

for i in temp_df.index:
    for j in temp_df.at[i, 'Paths'].split(' > '):
        channel.append(j)
        conversion.append(round(1/len(temp_df.at[i, 'Paths'].split(' > ')),0))
lin_att_df = pd.DataFrame()
lin_att_df['Channel'] = channel
lin_att_df['Conversion'] =conversion
lin_att_df = lin_att_df.groupby(['Channel']).sum().reset_index()
lin_att_df["Values"] = round(lin_att_df["Conversion"]/num_conv,0)
# print(lin_att_df["Conversion"])


# ## Shapley Value
shap_op=chnl_att.shapley_value(data_processed,unique_channels)


base_crs_tb =pd.crosstab(data_df["channel_name"],data_df["is_converted"]).reset_index().sort_values("channel_name")
base_impressions = base_crs_tb[0]

fst_crs_tb = df_frst.sort_values("Channel")
lst_crs_tb = df_lst.sort_values("Channel")
linear_crs_tab = lin_att_df.sort_values("Channel")


fst_crs_tb.rename(columns = {0:'Impression',1:"Conversion"}, inplace = True)
lst_crs_tb.rename(columns = {0:'Impression',1:"Conversion"}, inplace = True)
linear_crs_tab.rename(columns = {0:'Impression',1:"Conversion"}, inplace = True)

fst_crs_tb["Impression"]=base_impressions
lst_crs_tb["Impression"]=base_impressions
linear_crs_tab["Impression"]=base_impressions

fst_crs_tb["conv/imps"] = fst_crs_tb["Conversion"]/fst_crs_tb["Impression"]
lst_crs_tb["conv/imps"] = lst_crs_tb["Conversion"]/lst_crs_tb["Impression"]
linear_crs_tab["conv/imps"] = linear_crs_tab["Conversion"]/linear_crs_tab["Impression"]

fst_crs_tb["model_name"] = "First Touch"
lst_crs_tb["model_name"] = "Last Touch"
linear_crs_tab["model_name"] = "Linear Touch"

marko_crs_tb = marko_op.sort_values("Channel")
marko_crs_tb["Conversion"] = round(marko_crs_tb["Values"]*num_conv,0)
marko_crs_tb["Impression"]=base_impressions
marko_crs_tb["conv/imps"] = marko_crs_tb["Conversion"]/marko_crs_tb["Impression"]
marko_crs_tb["model_name"] = "Markov Chain"


shap_crs_tb = shap_op.sort_values("Channel")
shap_crs_tb["Conversion"] = round(shap_crs_tb["Values"]*num_conv,0)
shap_crs_tb["Impression"]=base_impressions
shap_crs_tb["conv/imps"] = shap_crs_tb["Conversion"]/shap_crs_tb["Impression"]
shap_crs_tb["model_name"] = "Shapley Value"

concat_df=pd.concat([fst_crs_tb,lst_crs_tb,linear_crs_tab,marko_crs_tb,shap_crs_tb])
# print(concat_df)

server = flask.Flask(__name__)

external_scripts = ['']
external_stylesheets = [
                        'https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',

                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',

                        ]

# Altimetrik logo image
# altimetrik_logo = r'D:\User\Marketing and Sales Analytics\Marketing channel attribution\MarketingChannelAttribution_12_05_2022\images\logo.png'
altimetrik_logo = 'images/img.png'
altimetrik_logo = base64.b64encode(open(altimetrik_logo, 'rb').read())



#Impressions_pie
data_impressions=pd.crosstab(data_df.channel_name, data_df.interaction)
data_impressions=data_impressions.reset_index()
impressions_pie=px.pie(data_impressions, values="impression",hole=0.3,names="channel_name", color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
impressions_pie.update_layout(margin=dict(t=0, b=0, l=20, r=20))
impressions_pie.update_layout(legend=dict(yanchor="top", y=1.09, x=0, xanchor="left", orientation="h"))
impressions_pie.update_layout({"plot_bgcolor":"rgba(0, 0, 0, 0)"})


##Conversions_Pie
conversions_pie=px.pie(data_impressions, values="conversion",names="channel_name",hole=0.3,color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
conversions_pie.update_layout(margin=dict(t=0, b=0, l=20, r=20))
conversions_pie.update_layout(legend=dict(yanchor="top", y=1.09, x=0, xanchor="left", orientation="h"))
conversions_pie.update_layout({"plot_bgcolor":"rgba(0, 0, 0, 0)"})

##Time plot

data_time_plot=data_df[['visit_time', 'channel_name', 'is_converted']].copy()
data_time_plot_conv=data_time_plot.loc[data_time_plot['is_converted']==1]
data_time_plot_conv["visit_time"]=pd.to_datetime(data_time_plot_conv['visit_time'], errors='coerce')
data_time_crosstab=pd.crosstab(data_time_plot_conv.visit_time, data_time_plot_conv.channel_name)
data_time_crosstab=data_time_crosstab.cumsum(axis=0).reset_index()
data_time_melt=pd.melt(data_time_crosstab, id_vars=['visit_time'])
data_melt_rank=data_time_melt.groupby(["visit_time"]).apply(lambda x:x.sort_values(by='value', ascending=False)).reset_index(drop=True)
data_melt_rank["Rank"]=data_melt_rank.groupby("visit_time")["value"].rank("dense", ascending=False)
data_melt_rank_final=data_melt_rank.groupby('visit_time').head(5).reset_index(drop=True)
fig = go.Figure()
for c in data_melt_rank_final['channel_name'].unique()[:20]:
    dfp = data_melt_rank_final[data_melt_rank_final['channel_name']==c].pivot(index='visit_time', columns='channel_name', values='Rank')
    fig.add_traces(go.Scatter(x=dfp.index, y=dfp[c], mode='markers+lines', name = c, marker_size=10, marker_symbol='hexagon2-open', marker_line_width=2))

fig.update_yaxes(autorange="reversed")
fig.update_layout(margin=dict(t=0, b=0, l=60, r=20))
fig.update_layout(legend=dict(yanchor="top", y=1.09, x=0, xanchor="left", orientation="h"))




tst = data_processed[data_processed["is_converted"]==1]

df = pd.DataFrame()
df["Channels"] =np.array(tst["Paths"].str.split(" > ",expand=True)).ravel()

df1= pd.DataFrame()
df1["Channel"] = df["Channels"].value_counts().keys()
df1["Count"] = df["Channels"].value_counts().values
df1["base_conv_rate"] = df1["Count"]/num_obs

channel_based_conv_pie=px.bar(df1, x="Channel",y="base_conv_rate",color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
channel_based_conv_pie.update_layout(margin=dict(t=0, b=50, l=20, r=20))
channel_based_conv_pie.update_layout({"plot_bgcolor":"rgba(0, 0, 0, 0)"})
channel_based_conv_pie.update_layout(yaxis_title="Base conversion rate")

nav_bar=dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src='data:image/png;base64,{}'.format(altimetrik_logo.decode()), height="30px",
                                         className=""), style={'paddingLeft': 0, 'paddingRight': 0}),
                        dbc.Col(dbc.NavbarBrand("Altimetrik", className="pl-0 pr-0",
                                                style={"font-size": 30, "color": "#ED7640",
                                                       'fontFamily': 'proxima-nova', "font-weight": "590"})),

                    ],
                    align="center",
                    className="pl-5",

                ),

            ),

        ],
        className=" fixed-top shadow-sm bg-white ",
        color="dark",
        dark=True,
        style={'paddingTop': 5, 'paddingBottom': 5,"height":"48px"},

    )


channel_info=html.Div([
    dbc.Row([
        dbc.Col([
            nav_bar
        ])
    ]),
    html.H4("Overview",style={"font-weight": "bold", "font-family": "proxima-nova", "font-size": "32px","marginLeft": "60px", "marginRight":"60px","marginTop":"80px"}, ),


    dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.A(id="total_users_id", children=[
                            html.H6("Total Leads:", style={"text-align": "center", "font-family": "proxima-nova"}),
                            html.H5(num_obs, id="total_use_id",
                                    style={"text-align": "center", "font-family": "proxima-nova", "font-size": 32}),
                        ])
                    ])
                ], className="shadow zoom pl-0", style={"backgroundColor": "#ADE3E6",})
            ], className="pl-0 pr-0 col-sm"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.A(id="total_conv_id", children=[
                            html.H6("Total Conversions:", style={"text-align": "center", "font-family": "proxima-nova"}),
                            html.H5(len(data_df[data_df["is_converted"] == 1]),
                                    style={"text-align": "center", "font-family": "proxima-nova", "font-size": 32}),
                        ])
                    ])
                ], className="shadow zoom", style={"backgroundColor": "#ADE3E6","marginLeft": "16px"})
            ], className="pl-0 pr-0 col-sm"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.A(id="total_uniq_channels_id", children=[
                            html.H6("Total Unique Channels:",
                                    style={"text-align": "center", "font-family": "proxima-nova"}),
                            html.H5(len(unique_channels),
                                    style={"text-align": "center", "font-family": "proxima-nova", "font-size": 32}),
                        ])
                    ])
                ], className="shadow zoom", style={"backgroundColor": "#ADE3E6","marginLeft": "16px"})
            ], className="pl-0 pr-0 col-sm"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.A(id="total_users_jour_id", children=[
                            html.H6("Base Conversion Rate:", style={"text-align": "center", "font-family": "proxima-nova"}),
                            html.H5(round((num_conv / num_obs), 2),
                                    style={"text-align": "center", "font-family": "proxima-nova", "font-size": 32}),
                        ])
                    ])
                ], className="shadow zoom", style={"backgroundColor": "#ADE3E6","marginLeft": "16px"})
            ], className="pl-0 pr-0 col-sm"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.A(id="total_cust_id", children=[
                        html.H6("New Customers (%):",
                                style={"text-align": "center", "font-family": "proxima-nova"}),
                        html.H5(round((num_conv/total_customers)*100, 2),
                                style={"text-align": "center", "font-family": "proxima-nova", "font-size": 32}),
                    ])
                ])
            ], className="shadow zoom", style={"backgroundColor": "#ADE3E6","marginLeft": "16px"})
        ], className="pl-0 pr-0 col-sm")
    ],style={"marginLeft": "60px", "marginRight": "60px","marginTop":"20px"}),




    dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Row(
                        dbc.Col([
                            html.H4("Impressions per channel",style={"font-family": "proxima-nova","font-weight": "bold",})
                        ], className="col-sm-12"), className="ml-3 mt-2"),
            ],className="shadow")
        ],className="pl-0 pr-0 col-sm-4"),
            dbc.Col([
                dbc.Card([
                    dbc.Row(
                        dbc.Col([
                            html.H4("Conversions per channel",style={"font-family": "proxima-nova","font-weight": "bold",})
                        ], className="col-sm-12"), className="ml-3 mt-2"),
                ],className="shadow",style={"marginLeft": "16px"})
            ],className="pl-0 pr-0 col-sm-4"),
            dbc.Col([
                dbc.Card([
                    dbc.Row(
                        dbc.Col([
                            html.H4("Channel-wise base conversions",style={"font-family": "proxima-nova","font-weight": "bold",})
                        ], className="col-sm-12"), className="ml-3 mt-2"),
                ], className="shadow",style={"marginLeft": "16px"})
            ], className="pl-0 pr-0 col-sm-4")
        ],style={"marginLeft": "60px", "marginRight": "60px","marginTop":"40px"}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                 dbc.Row([
                    dbc.Col([
                        dcc.Loading(dcc.Graph(id="impressions_plt",figure=impressions_pie,config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),
                                    type="default",color="#2B3348"),

                     ],),
                     ])
            ],className="shadow")
        ],className="pl-0 pr-0 col-sm-4"),
        dbc.Col([
            dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(dcc.Graph(id="conversions_plt",figure=conversions_pie,config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),
                                        type="default",color="#2B3348"),

                        ],),
                     ])
            ],className="shadow",style={"marginLeft": "16px"})
        ],className="pl-0 pr-0 col-sm-4"),
        dbc.Col([
            dbc.Card([
                dbc.Row([
                    dbc.Col([
                            dcc.Loading(dcc.Graph(id="channel_based_conv_plt",figure=channel_based_conv_pie,config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),type="default",color="#2B3348"),

                     ],),
                ])
            ], className="shadow",style={"marginLeft": "16px"})
        ], className="pl-0 pr-0 col-sm-4")
    ],style={"marginLeft": "60px", "marginRight": "60px"}),

    dbc.Row([
        dbc.Col([
            html.H4('Attribution model',
                    style={"font-family": "proxima-nova", "font-size": '32px', "font-weight": "bold", })
        ],className="pl-0 pr-0 col-sm-8")
    ],style={"marginLeft": "60px", "marginRight": "60px", "marginTop": "40px"}),
    dbc.Row([
        dbc.Col([
            dbc.Collapse([
                html.H5('First touch attribution',id="model_name",style={"font-family": "proxima-nova", "font-weight": "bold",})
            ],id="model_name_collapse",is_open=True),
        ],className="pl-0 pr-0 col-sm-8"),

        dbc.Col([
            dbc.ButtonGroup(
                [
                    dbc.DropdownMenu(
                        [dbc.DropdownMenuItem("First Touch",id="first_touch_btn",n_clicks=0),
                         dbc.DropdownMenuItem("Last Touch",id="last_touch_btn",n_clicks=0),
                         dbc.DropdownMenuItem("Linear Touch",id="linear_touch_btn",n_clicks=0),
                         dbc.DropdownMenuItem("Shapley Value",id="shapley_btn",n_clicks=0),
                         dbc.DropdownMenuItem("Markov Chain",id="markov_btn",n_clicks=0),
                         ],
                        label="Select Model",
                        group=True,
                        direction="down",
                        id="model_select_id",
                    ),
                    dbc.Button("Compare",id='compare_button_id',n_clicks=0),

                ],className="p-0",style={"marginLeft":"40%"})
        ],className="pl-0 pr-0 col-sm-4")
    ],style={"marginLeft": "60px", "marginRight": "60px"}),


    dbc.Collapse([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Attribution",id="chart_title_id",style={"font-family": "proxima-nova","font-weight": "bold",})
                        ],className="col-sm-6"),
                            dbc.Collapse([
                                dbc.Col([
                                    html.Div([
                                        dbc.Label(className="switch btn-color-mode-switch", children=[
                                            dbc.Input(type="checkbox", name="color_mode", id="color_mode", value=1, step=1,
                                                      persistence=True),
                                            dbc.Label(html_for="color_mode", className="btn-color-mode-switch-inner",
                                                      id="lable_name", check=True),

                                        ])
                                    ], className="btn-container mb-1 p-0", id="div_click", n_clicks=1,),

                        ], className="col-sm-6 p-0",style={"marginLeft":"25%"}),

                        ],id="toggel_collapse",is_open=False),
                    ],className="ml-3 mt-2"),
                ], className="shadow")
            ], className="pl-0 pr-0 col-sm-6"),
            dbc.Col([
                dbc.Card([
                    dbc.Row(
                        dbc.Col([
                            html.H4("Conversion rate",style={"font-family": "proxima-nova", "font-weight": "bold",})
                        ], className="col-sm-12"),className="ml-3 mt-2"),
                ], className="shadow",style={"marginLeft": "16px"})
            ], className="pl-0 pr-0 col-sm-6"),

        ], style={"marginLeft": "60px", "marginRight": "60px","marginTop":"20px"}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(dcc.Graph(id="attr_pie_plot",config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),type="default",color="#2B3348"),

                        ],className="col-sm-12")
                    ])

                ], className="shadow")
            ], className="pl-0 pr-0 col-sm-6"),
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(dcc.Graph(id="imprs_conv_bar_plot",config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),type="default",color="#2B3348"),

                        ],className="col-sm-12" ),
                    ])
                ], className="shadow",style={"marginLeft": "16px"})
            ], className="pl-0 pr-0 col-sm-6"),
        ], style={"marginLeft": "60px", "marginRight": "60px",}),
        dbc.Collapse([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.Row([
                            dbc.Col([
                                html.H4("Transitions in markov attribution model",
                                        style={"font-family": "proxima-nova", "font-weight": "bold", })
                            ], className="col-sm-9 p-0"),
                            dbc.Col([
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("Heat Map", outline=True, color="primary", id='heat_map_id',n_clicks=1),
                                        dbc.Button("Bar chart", outline=True, color="primary", id='bar_chart_id'),
                                    ]
                                )
                            ], className="col-sm-3 mb-1"),
                        ], className="ml-3 mt-2"),
                    ], className="shadow")
                ], className="pl-0 pr-0 col-sm-12")

            ], style={"marginLeft": "60px", "marginRight": "60px", "marginTop": "40px"}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.Row([
                            dbc.Col([
                                dcc.Loading(dcc.Graph(id="heatmap_plot", figure=heatmap_marko,config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),type="default",color="#2B3348"),

                            ], ),
                        ]),
                        dbc.Collapse(
                            dbc.Row([
                                dbc.Col([
                                ],className="col-sm-5"),
                                dbc.Col([
                                    html.P(className="pl-0",style={"font-family": "proxima-nova",'font-size': '14px',"marginLeft":"8%"},id="note_text")
                                ],),
                            ]),id="note_collapse",is_open=False,className="ml-3 mt-3 mb-3"),
                    ], className="pl-0 shadow")
                ], className="pl-0 pr-0 col-sm-12"),

        ], style={"marginLeft": "60px", "marginRight": "60px", }),
        ],id="heatmap_collapse",is_open=False)
    ],id="model_collapse",is_open=True),

    dbc.Collapse([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Comparison of attribution models",style={"font-family": "proxima-nova","font-weight": "bold",})
                        ],className="col-sm-6"),
                        dbc.Col([
                            dcc.Checklist(
                                ['First Touch', 'Last Touch', 'Linear Touch','Shapley Value','Markov Chain'],
                                ['First Touch', 'Last Touch', 'Linear Touch','Shapley Value','Markov Chain'],
                                inline=True,id="check_list_id",labelStyle = {'display': 'block', 'cursor': 'pointer', 'margin-left':'20px'},style={"font-family": "proxima-nova","font-weight": "bold",}),
                        ],className="col-sm-6")
                    ],className="ml-3 mt-2"),

                ], className="shadow")
            ], className="pl-0 pr-0 col-sm-12"),
        ], style={"marginLeft": "60px", "marginRight": "60px", "marginTop": "20px"}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(dcc.Graph(id="compare_value_plot",config={
                                        "displaylogo": False,'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'ZoomIn', 'ZoomOut',
                                       'Zoom', 'autoscale','toImageButtonOptions']}),type="default",color="#2B3348"),

                        ],className="col-sm-12")
                    ]),

                ], className="shadow")
            ], className="pl-0 pr-0 col-sm-12"),
        ], style={"marginLeft": "60px", "marginRight": "60px",}),

    ],id="compare_collapse",is_open=False),

])



app = dash.Dash(__name__,server=server, external_scripts=external_scripts, external_stylesheets=external_stylesheets,url_base_pathname='/')
app.layout=html.Div(children=[
                              channel_info,
                              html.Div([
                                    html.B(["© 2022 Altimetrik Corp."], style={"height":"40px","marginLeft": "50%", "color": "#5C7191", "fontSize": 13,"font-family":"proxima-nova"}),
                                ], style={"background-color": "#F7F7F7", "padding": 10}),
                            ],
                    style={"backgroundColor":"#F5F5F5"})

flag = 2
tog_prev=1
pre_lis = [0,0,0,0,0,0]
@app.callback(
    [
        Output(component_id="attr_pie_plot",component_property="figure"),
        Output(component_id="imprs_conv_bar_plot",component_property="figure"),
        Output(component_id="first_touch_btn",component_property="n_clicks"),
        Output(component_id="last_touch_btn",component_property="n_clicks"),
        Output(component_id="linear_touch_btn",component_property="n_clicks"),
        Output(component_id="shapley_btn",component_property="n_clicks"),
        Output(component_id="markov_btn",component_property="n_clicks"),
        Output(component_id="heatmap_collapse", component_property="is_open"),
        Output(component_id="model_collapse",component_property="is_open"),
        Output(component_id="compare_collapse",component_property="is_open"),
        Output(component_id="compare_button_id", component_property="n_clicks"),
        Output(component_id="compare_value_plot", component_property="figure"),
        Output(component_id="check_list_id", component_property="value"),
        Output(component_id="toggel_collapse",component_property="is_open"),
        Output(component_id="div_click",component_property="n_clicks"),
        Output(component_id="model_name",component_property="children"),
        Output(component_id="chart_title_id",component_property="children"),
        Output(component_id="model_name_collapse",component_property="is_open"),
        Output(component_id="heatmap_plot",component_property="figure"),
        Output(component_id="heat_map_id",component_property="n_clicks"),
        Output(component_id="bar_chart_id",component_property="n_clicks"),
        Output(component_id="note_collapse",component_property="is_open"),
        Output(component_id="note_text",component_property="children"),

    ],
    [
        Input(component_id="first_touch_btn",component_property="n_clicks"),
        Input(component_id="last_touch_btn",component_property="n_clicks"),
        Input(component_id="linear_touch_btn",component_property="n_clicks"),
        Input(component_id="shapley_btn",component_property="n_clicks"),
        Input(component_id="markov_btn",component_property="n_clicks"),
        Input(component_id="compare_button_id",component_property="n_clicks"),
        Input(component_id="check_list_id", component_property="value"),
        Input(component_id="div_click",component_property="n_clicks"),
        Input(component_id="model_name",component_property="children"),
        Input(component_id="chart_title_id",component_property="children"),
        Input(component_id="heat_map_id",component_property="n_clicks"),
        Input(component_id="bar_chart_id",component_property="n_clicks"),
        Input(component_id="note_text",component_property="children"),


    ],
    [
        State(component_id="heatmap_collapse",component_property="is_open"),
        State(component_id="model_collapse",component_property="is_open"),
        State(component_id="compare_collapse",component_property="is_open"),
        State(component_id="toggel_collapse",component_property="is_open"),
        State(component_id="model_name_collapse",component_property="is_open"),
        State(component_id="note_collapse",component_property="is_open"),
    ]
)
def model_callbacks(frst,last,lnr,shap,mark,comp_btn,check_list,div_click,modl_name,chart_title_id,heat_map,bar_chart,note_text,is_open,mdl_collapse,com_collapse,toggel_collapse,model_name_collapse,note_collapse):
    global tog_prev, flag,pre_lis
    note_collapse=False
    curr_lis=[frst,last,lnr,shap,mark,comp_btn]


    if heat_map:
        marko_mtrx_1 = marko_mtrx.drop("Start",axis=1)
        heatmap_marko = px.imshow(marko_mtrx_1, text_auto=True, aspect='auto',
                                  color_continuous_scale=px.colors.sequential.Blues)
        heatmap_marko.update_layout(yaxis_title="Channels")
        note_collapse = True
        note_text="Note : Row entries are the probabilities of transitioning from row heading to each column heading"

    elif bar_chart:
        marko_mtrx_1 = marko_mtrx.reset_index()
        marko_mtrx_1 =marko_mtrx_1[~marko_mtrx_1["index"].isin([ 'NULL', 'Start','Conversion'])]
        heatmap_marko = px.bar(marko_mtrx_1, x="index", y=['Facebook', 'Instagram', 'Online Display', 'Online Video',
                                                           'Paid Search', 'Conversion', 'NULL', 'Start'],
                               barmode='group',
                               color_discrete_sequence=["#51C8CF", "#FE7519", "#767B88", "#FCD15D", "#1976D2",
                                                        "#232A3D",
                                                        "#B38269", "#1976D2"])

        heatmap_marko.update_layout(legend=dict(yanchor="top", y=1.09, x=0, xanchor="left", orientation="h"))
        heatmap_marko.update_layout(yaxis_title="Probability", xaxis_title="Channels")
        note_collapse =True
        note_text="Note : The height of bars correspond to probabilities of transitioning from category to each legend entry"


    else:
        marko_mtrx_1 = marko_mtrx.drop("Start", axis=1)
        heatmap_marko = px.imshow(marko_mtrx_1, text_auto=True, aspect='auto',
                                  color_continuous_scale=px.colors.sequential.Blues)
        heatmap_marko.update_layout(yaxis_title="Channels")
        note_collapse = True
        note_text="Note : Row entries are the probabilities of transitioning from row heading to each column heading"
    heatmap_marko.update_layout(margin=dict(t=10, b=10, l=30, r=10))
    heatmap_marko.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)"})
    if pre_lis==curr_lis:
        pre_lis=curr_lis
    else:
        pre_lis=[i^j for i,j in zip(curr_lis,pre_lis)]

    if pre_lis[0]:
        df_frst =concat_df[concat_df["model_name"]=="First Touch"]
        model_attr_pie = px.pie(df_frst, values="Conversion", hole=0.5,names="Channel",color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
        model_attr_bar = px.bar(df_frst, x='Channel', y='conv/imps',color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
        is_open=False
        mdl_collapse = True
        com_collapse = False
        toggel_collapse = False
        mdl_name_collapse = True
        modl_name = "First touch attribution"
        chart_title_id="Attribution"

    elif pre_lis[1]:
        df_lst = concat_df[concat_df["model_name"] == "Last Touch"]
        model_attr_pie = px.pie(df_lst, values="Conversion", hole=0.5,  names="Channel",color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
        model_attr_bar = px.bar(df_lst, x='Channel', y='conv/imps',color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
        is_open = False
        mdl_collapse = True
        com_collapse = False
        toggel_collapse = False
        model_name_collapse = True
        modl_name = "Last touch attribution"
        chart_title_id = "Attribution"
    elif pre_lis[2]:
        df_linear = concat_df[concat_df["model_name"] == "Linear Touch"]
        model_attr_pie = px.pie(df_linear, values="Conversion", hole=0.5, names="Channel",color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
        model_attr_bar = px.bar(df_linear, x='Channel', y='conv/imps',color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
        is_open = False
        mdl_collapse = True
        com_collapse = False
        toggel_collapse = False
        model_name_collapse = True
        modl_name = "Linear touch attribution"
        chart_title_id = "Attribution"
    elif pre_lis[3]:
        df_shap = concat_df[concat_df["model_name"] == "Shapley Value"]
        model_attr_pie = px.pie(df_shap, values="Values", hole=0.5,  names="Channel",color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
        model_attr_bar = px.bar(df_shap, x='Channel', y='conv/imps',color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
        is_open = False
        mdl_collapse = True
        com_collapse = False
        toggel_collapse = True
        model_name_collapse = True
        modl_name = "Shapley value"
        if tog_prev != div_click:
            flag = flag+1
            tog_prev = div_click
        else:
            pass
        if flag%2==0:
            model_attr_pie = px.pie(df_shap, values="Conversion", hole=0.5,
                                    names="Channel",
                                    color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
            chart_title_id = "Attribution"
        else:
            model_attr_pie = px.bar(df_shap, x="Channel",y="Values",color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
            chart_title_id = "Values"
    elif pre_lis[4]:
        df_markov = concat_df[concat_df["model_name"] == "Markov Chain"]
        model_attr_bar = px.bar(df_markov, x='Channel', y='conv/imps',color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
        is_open = True
        mdl_collapse = True
        com_collapse = False
        toggel_collapse=True
        model_name_collapse = True
        modl_name = "Markov chain attribution"

        if tog_prev != div_click:
            flag = flag+1
            tog_prev = div_click
        else:
            pass
        if flag%2==0:
            model_attr_pie = px.pie(df_markov, values="Conversion", hole=0.5,
                                    names="Channel",
                                    color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
            chart_title_id = "Attribution"
        else:
            model_attr_pie = px.bar(df_markov, x="Channel",y="Values",color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
            chart_title_id = "Values"
    else:
        df_frst = concat_df[concat_df["model_name"] == "First Touch"]
        model_attr_pie = px.pie(df_frst, values="Conversion", hole=0.5,  names="Channel",color_discrete_sequence=["#FE7519","#767B88","#FCD15D","#1976D2"] )
        model_attr_bar = px.bar(df_frst, x='Channel', y='conv/imps',color=["#51C8CF","#FE7519","#767B88","#FCD15D"],color_discrete_map="identity")
        is_open = False
        toggel_collapse =False
        model_name_collapse = True
        modl_name = "First touch attribution"
        chart_title_id = "Attribution"

    if pre_lis[5]:
        models_df =concat_df[concat_df["model_name"].isin(check_list)]

        compare_value_plot = tls.make_subplots(rows=2, cols=1, vertical_spacing=0.06, )

        fig1 = px.bar(models_df, x="model_name", y="Conversion",
                      color='Channel', barmode='group',
                      color_discrete_sequence=["#51C8CF", "#FE7519", "#767B88", "#FCD15D"])
        fig1["data"][0].showlegend = False
        fig1["data"][1].showlegend = False
        fig1["data"][2].showlegend = False
        fig1["data"][3].showlegend = False
        fig1["data"][4].showlegend = False

        compare_value_plot.append_trace(fig1["data"][0], 1, 1)
        compare_value_plot.append_trace(fig1["data"][1], 1, 1)
        compare_value_plot.append_trace(fig1["data"][2], 1, 1)
        compare_value_plot.append_trace(fig1["data"][2], 1, 1)
        compare_value_plot.append_trace(fig1["data"][3], 1, 1)
        compare_value_plot.append_trace(fig1["data"][4], 1, 1)

        fig2 = px.bar(models_df, x="model_name", y="conv/imps",
                      color='Channel', barmode='group',
                      color_discrete_sequence=["#51C8CF", "#FE7519", "#767B88", "#FCD15D"])

        compare_value_plot.append_trace(fig2["data"][0], 2, 1)
        compare_value_plot.append_trace(fig2["data"][1], 2, 1)
        compare_value_plot.append_trace(fig2["data"][2], 2, 1)
        compare_value_plot.append_trace(fig2["data"][3], 2, 1)
        compare_value_plot.append_trace(fig2["data"][4], 2, 1)

        compare_value_plot['layout']['yaxis']['title'] = 'Attributed conversions'
        compare_value_plot['layout']['yaxis2']['title'] = 'Conversion rates'

        compare_value_plot.update_layout(margin=dict(t=10, b=10, l=30, r=10))
        compare_value_plot.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)"})
        compare_value_plot.update_layout(legend_title_text=None)


        mdl_collapse =False
        com_collapse=True
        model_name_collapse = False
        compare_value_plot.update_layout(legend=dict(yanchor="top", y=1.09, x=0, xanchor="left", orientation="h"))


    else:
        compare_value_plot = px.bar(concat_df, x="model_name", y="Conversion",
                               color='Channel', barmode='group', color_discrete_sequence=["#51C8CF","#FE7519","#767B88","#FCD15D"])
        compare_value_plot.update_layout(margin=dict(t=10, b=10, l=30, r=10))
        compare_value_plot.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)"})
        model_name_collapse = True
        compare_value_plot.update_layout(legend=dict(yanchor="top", y=1.09, x=0, xanchor="left", orientation="h"))


    model_attr_pie.update_layout(margin=dict(t=10, b=10, l=15, r=10))
    model_attr_pie.update_layout(legend=dict(x=0.37, y=0.5))

    model_attr_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10))


    [frst, last, lnr, shap, mark,comp_btn] = pre_lis
    model_attr_bar.update_layout({"plot_bgcolor":"rgba(0, 0, 0, 0)"})
    model_attr_pie.update_layout({"plot_bgcolor":"rgba(0, 0, 0, 0)"})
    model_attr_pie.update_layout(yaxis_title="Importance")
    model_attr_bar.update_layout(yaxis_title="Conversion rate")



    return model_attr_pie,model_attr_bar,frst, last, lnr, shap, mark, is_open,mdl_collapse,com_collapse,comp_btn,compare_value_plot,check_list,toggel_collapse,div_click,modl_name,chart_title_id,model_name_collapse,heatmap_marko,0,0,note_collapse,note_text




if __name__ == '__main__':
    app.run_server(port=8000,debug=True)