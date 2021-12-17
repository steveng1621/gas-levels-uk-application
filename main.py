import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import date
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# In[4]:


df1 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\Nitrogen formatted part 1.csv')
df2 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\Nitrogen formatted part 2.csv')
df3 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\Nitrogen formatted part 3.csv')
df4 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\OZone formatted part 1.csv')
df5 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\PM2 formatted part 1.csv')
df6 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\PM2 formatted part 2.csv')
df7 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\PM10 formatted part 1.csv')
df8 = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\PM10 formatted part 2.csv')


# In[5]:


def BuildDataframe(dataframe):
    col_i = 0
    array = []
    location = ""
    for col in dataframe.columns:
        if col.find("Unnamed:") != 0:
            location = col
        elif col.find("Unnamed: 0") == 0 or col.find("Unnamed: 1") == 0:
            location = dataframe.loc[0][col_i]
        array.append([location, dataframe.loc[0][col_i]])
        col_i += 1
    header = pd.MultiIndex.from_tuples(array, names=['loc', 'value'])
    dataframe.columns = header
    dataframe = dataframe.iloc[1:, :]
    return dataframe


# In[6]:


df1_formatted = BuildDataframe(df1)
df2_formatted = BuildDataframe(df2)
df3_formatted = BuildDataframe(df3)
df4_formatted = BuildDataframe(df4)
df5_formatted = BuildDataframe(df5)
df6_formatted = BuildDataframe(df6)
df7_formatted = BuildDataframe(df7)
df8_formatted = BuildDataframe(df8)

# In[7]:


df = pd.concat([df1_formatted, df2_formatted, df3_formatted, df4_formatted, df5_formatted, df6_formatted, df7_formatted,
                df8_formatted], axis=1)
df = df.T.drop_duplicates().T

# In[8]:


Nitrogen_dioxide = df.xs('Nitrogen dioxide', level='value', axis=1)
Carbon_monoxide = df.xs('Carbon monoxide', level='value', axis=1)
PM2_5_particulate = df.xs('PM2.5 particulate matter (Hourly measured)', level='value', axis=1)
PM10_particulat = df.xs('PM10 particulate matter (Hourly measured)', level='value', axis=1)
Ozone = df.xs('Ozone', level='value', axis=1)

# In[9]:


Nitrogen_dioxide['Date'] = df['Date']['Date']
Carbon_monoxide['Date'] = df['Date']['Date']
PM2_5_particulate['Date'] = df['Date']['Date']
PM10_particulat['Date'] = df['Date']['Date']
Ozone['Date'] = df['Date']['Date']
Nitrogen_dioxide['Time'] = df['Time']['Time']
Carbon_monoxide['Time'] = df['Time']['Time']
PM2_5_particulate['Time'] = df['Time']['Time']
PM10_particulat['Time'] = df['Time']['Time']
Ozone['Time'] = df['Time']['Time']
Nitrogen_dioxide['Type_name'] = "Nitrogen Dioxide (NO2)"
Carbon_monoxide['Type_name'] = "Carbon Monoxide CO"
PM2_5_particulate['Type_name'] = "Particulate Matter 2.5 (PM 2.5)"
PM10_particulat['Type_name'] = "Particulate Matter 10 (PM 10)"
Ozone['Type_name'] = "Ozone O3"
Nitrogen_dioxide['Type'] = "NO2"
Carbon_monoxide['Type'] = "CO"
PM2_5_particulate['Type'] = "PM 2.5"
PM10_particulat['Type'] = "PM 10"
Ozone['Type'] = "O3"

# In[10]:


Nitrogen_dioxide = Nitrogen_dioxide.replace('No data', np.nan)
Carbon_monoxide = Carbon_monoxide.replace('No data', np.nan)
PM2_5_particulate = PM2_5_particulate.replace('No data', np.nan)
PM10_particulat = PM10_particulat.replace('No data', np.nan)
Ozone = Ozone.replace('No data', np.nan)


# In[11]:


def BuildGraphData(dataframe):
    cols = [i for i in dataframe.columns if i not in ["Type", "Time", 'Type_name']]
    for col in cols:
        if col == 'Date':
            dataframe[col] = pd.to_datetime(dataframe[col], utc=False)
        else:
            dataframe[col] = pd.to_numeric(dataframe[col])
    return dataframe


# In[12]:


Nitrogen_dioxide = BuildGraphData(Nitrogen_dioxide)
Carbon_monoxide = BuildGraphData(Carbon_monoxide)
PM2_5_particulate = BuildGraphData(PM2_5_particulate)
PM10_particulat = BuildGraphData(PM10_particulat)
Ozone = BuildGraphData(Ozone)

# In[13]:


formated_Nitrogen_dioxide = Nitrogen_dioxide.set_index("Date")
formated_Carbon_monoxide = Carbon_monoxide.set_index("Date")
formated_PM2_5_particulate = PM2_5_particulate.set_index("Date")
formated_PM10_particulat = PM10_particulat.set_index("Date")
formated_Ozone = Ozone.set_index("Date")

# In[14]:


graph_Nitrogen_dioxide = Nitrogen_dioxide.set_index("Date").select_dtypes(np.number).stack().groupby(level=0).describe()
graph_Carbon_monoxide = Carbon_monoxide.set_index("Date").select_dtypes(np.number).stack().groupby(level=0).describe()
graph_PM2_5_particulate = PM2_5_particulate.set_index("Date").select_dtypes(np.number).stack().groupby(
    level=0).describe()
graph_PM10_particulat = PM10_particulat.set_index("Date").select_dtypes(np.number).stack().groupby(level=0).describe()
graph_Ozone = Ozone.set_index("Date").select_dtypes(np.number).stack().groupby(level=0).describe()

# In[15]:


mask_graph_Nitrogen_dioxide = (graph_Nitrogen_dioxide.index >= '2021-03-01') & (
            graph_Nitrogen_dioxide.index < '2021-06-01')
mask_graph_Carbon_monoxide = (graph_Carbon_monoxide.index >= '2021-03-01') & (
            graph_Carbon_monoxide.index < '2021-06-01')
mask_graph_PM2_5_particulate = (graph_PM2_5_particulate.index >= '2021-03-01') & (
            graph_PM2_5_particulate.index < '2021-06-01')
mask_graph_PM10_particulat = (graph_PM10_particulat.index >= '2021-03-01') & (
            graph_PM10_particulat.index < '2021-06-01')
mask_graph_Ozone = (graph_Ozone.index >= '2021-03-01') & (graph_Ozone.index < '2021-06-01')

mask_formatted_Nitrogen_dioxide = (formated_Nitrogen_dioxide.index >= '2021-03-01') & (
            formated_Nitrogen_dioxide.index < '2021-06-01')
mask_formatted_Carbon_monoxide = (formated_Carbon_monoxide.index >= '2021-03-01') & (
            formated_Carbon_monoxide.index < '2021-06-01')
mask_formatted_PM2_5_particulate = (formated_PM2_5_particulate.index >= '2021-03-01') & (
            formated_PM2_5_particulate.index < '2021-06-01')
mask_formatted_PM10_particulat = (formated_PM10_particulat.index >= '2021-03-01') & (
            formated_PM10_particulat.index < '2021-06-01')
mask_formatted_Ozone = (formated_Ozone.index >= '2021-03-01') & (formated_Ozone.index < '2021-06-01')

# In[16]:


filtered_graph_Nitrogen_dioxide = graph_Nitrogen_dioxide.loc[mask_graph_Nitrogen_dioxide]
filtered_graph_Carbon_monoxide = graph_Carbon_monoxide.loc[mask_graph_Nitrogen_dioxide]
filtered_graph_PM2_5_particulate = graph_PM2_5_particulate.loc[mask_graph_PM2_5_particulate]
filtered_graph_PM10_particulat = graph_PM10_particulat.loc[mask_graph_PM10_particulat]
filtered_graph_Ozone = graph_Ozone.loc[mask_graph_Ozone]

filtered_formated_Nitrogen_dioxide = formated_Nitrogen_dioxide.loc[mask_formatted_Nitrogen_dioxide]
filtered_formated_Carbon_monoxide = formated_Carbon_monoxide.loc[mask_formatted_Carbon_monoxide]
filtered_formated_PM2_5_particulate = formated_PM2_5_particulate.loc[mask_formatted_PM2_5_particulate]
filtered_formated_PM10_particulat = formated_PM10_particulat.loc[mask_formatted_PM10_particulat]
filtered_formated_Ozone = formated_Ozone.loc[mask_formatted_Ozone]

# In[17]:


filtered_formated_Nitrogen_dioxide['Date'] = filtered_formated_Nitrogen_dioxide.index
pivot_filtered_formated_Nitrogen_dioxide = filtered_formated_Nitrogen_dioxide.melt(
    id_vars=["Time", "Type", "Type_name", "Date"], var_name="loc", value_name="Value")
filtered_formated_Carbon_monoxide['Date'] = filtered_formated_Carbon_monoxide.index
pivot_filtered_formated_Carbon_monoxide = filtered_formated_Carbon_monoxide.melt(
    id_vars=["Time", "Type", "Type_name", "Date"], var_name="loc", value_name="Value")
filtered_formated_PM2_5_particulate['Date'] = filtered_formated_PM2_5_particulate.index
pivot_filtered_formated_PM2_5_particulate = filtered_formated_PM2_5_particulate.melt(
    id_vars=["Time", "Type_name", "Type", "Date"], var_name="loc", value_name="Value")
filtered_formated_PM10_particulat['Date'] = filtered_formated_PM10_particulat.index
pivot_filtered_formated_PM10_particulate = filtered_formated_PM10_particulat.melt(
    id_vars=["Time", "Type_name", "Type", "Date"], var_name="loc", value_name="Value")
filtered_formated_Ozone['Date'] = filtered_formated_Ozone.index
pivot_filtered_formated_Ozone = filtered_formated_Ozone.melt(id_vars=["Time", "Type_name", "Type", "Date"],
                                                             var_name="loc", value_name="Value")

# In[18]:


all_data_unformatted = pd.concat([pivot_filtered_formated_Nitrogen_dioxide, pivot_filtered_formated_Carbon_monoxide,
                                  pivot_filtered_formated_PM2_5_particulate, pivot_filtered_formated_PM10_particulate,
                                  pivot_filtered_formated_Ozone])

# In[19]:


settlement_types = pd.read_excel(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\CBP-8322.xlsx', sheet_name="Sheet1")
settlement_types.rename(columns={
    'Local authority name': 'Location_name',
    'Summary classification\n(Largest category. Orange shading indicates less than 50% of population in this category)'
    :'settlement_types'},
                        inplace=True)
settlement_types_lookup = settlement_types[['Location_name', 'settlement_types', 'region']]

locations_metadata = pd.read_csv(
    r'C:\Users\steve\PycharmProjects\pythonProject\pythonProject\gas-levels-uk\data\location_metedata.csv')
locations_metadata_lookup = locations_metadata[
    ['Site Name', 'Zone', 'Latitude', 'Longitude', 'lookupvalue_data', 'lookup_settelment']]
locations_metadata_lookup = locations_metadata_lookup.merge(settlement_types_lookup, left_on='lookup_settelment',
                                                            right_on='Location_name', how='left')

# In[20]:


all_data = all_data_unformatted.merge(locations_metadata_lookup, left_on='loc', right_on='lookupvalue_data', how='left')
all_data = all_data[all_data["settlement_types"].isnull() == False]
settlements_types_group = {'Core City': 'City', 'Large Town': 'Town', 'Medium Town': 'Town', 'Other City': 'City',
                           'Town': 'Town', 'Village or smaller': 'Village'}
all_data["settlement_types"] = all_data["settlement_types"].map(settlements_types_group)
all_data = all_data[all_data['Type'].isin(['NO2', 'PM 2.5', 'PM 10'])]

# In[21]:


dayOfWeek = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
all_data['weekday_no'] = all_data['Date'].dt.dayofweek
all_data['weekday'] = all_data['Date'].dt.dayofweek.map(dayOfWeek)

##all_data['Time'] = pd.to_datetime(all_data['Time'].replace("24:00:00","00:00:01"),format="%H:%M:%S")


# In[22]:


all_data = all_data[(all_data["settlement_types"].isnull() != True)]
all_data_all = all_data.copy()
all_data_all['Type'] = 'All'
all_data = [all_data, all_data_all]
all_data = pd.concat(all_data)

# In[ ]:




# ---------------------------------------------------------------
app = dash.Dash(__name__)
# ---------------------------------------------------------------
import plotly
import plotly.express as px

COLORS = px.colors.qualitative.Plotly
# ---------------------------------------------------------------
from sklearn import preprocessing, cluster

# ---------------------------------------------------------------


# ---------------------------------------------------------------
df = all_data
regions = sorted(list(all_data[all_data['region'].isnull() == False]['region'].astype(str).unique()))
# Types = sorted(list(all_data['Type'].astype(str).unique()))
settlements = sorted(list(all_data['settlement_types'].astype(str).unique()))

# ---------------------------------------------------------------
# ---------------------------------------------------------------
options_chart = []
options_chart.append({'label': 'Avg Hourly Gas Levels for the UK', 'value': 'Avg Hourly Gas Levels for the UK'})
options_chart.append({'label': 'Avg Hourly Gas Levels by Region', 'value': 'Avg Hourly Gas Levels by Region'})
options_chart.append({'label': 'Avg Hourly Gas Levels by Settlement', 'value': 'Avg Hourly Gas Levels by Settlement'})
options_chart.append({'label': 'Avg Hourly Gas Levels by Week Day', 'value': 'Avg Hourly Gas Levels by Week Day'})
options_chart.append({'label': 'Avg Hourly Gas Levels by Hour', 'value': 'Avg Hourly Gas Levels by Hour'})
# ---------------------------------------------------------------
# ---------------------------------------------------------------
options_regions = []

for region in regions:
    options_regions.append({'label': region, 'value': region})
# ---------------------------------------------------------------
# ---------------------------------------------------------------
options_type = []
options_type.append({'label': 'All', 'value': "All"})
options_type.append({'label': 'Nitrogen Dioxide (NO2)', 'value': "NO2"})
options_type.append({'label': 'Particulate Matter 2.5 (PM 2.5)', 'value': "PM 2.5"})
options_type.append({'label': 'Particulate Matter 10 (PM 10)', 'value': "PM 10"})
# ---------------------------------------------------------------
# ---------------------------------------------------------------
options_settlements = []

for settlement in settlements:
    options_settlements.append({'label': settlement, 'value': settlement})
# ---------------------------------------------------------------
# ---------------------------------------------------------------
options_loc = []
for location in sorted(list(all_data['loc'].astype(str).unique())):
    options_loc.append({'label': location, 'value': location})
# ---------------------------------------------------------------
# ---------------------------------------------------------------
app.layout = html.Div([

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='nine columns'),

    html.Div([

        html.Br(),
        html.Div([
            html.H4('Unit of measure:'),
            html.Label(["(ug/m3): Micrograms per Cubic Meter of Air"]),
            html.H4('Gas Types:'),
            html.Label(["(NO2): Nitrogen Dioxide"]),
            html.Br(),
            html.Label(["(PM 2.5): Particulate Matter 2.5"]),
            html.Br(),
            html.Label(["(PM 10): Particulate Matter 10"]),
            html.Br(),
            html.A('Click for More details', target="_blank",
                   href='https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health')

        ], id='output_chart'
        ),

        html.Div([
            html.H4('Box Plot Usage:'),
            html.Img(style={'height': '90%', 'width': '100%'}, src=app.get_asset_url('box_plot.jpg')),

        ], id="img_box_plot", style={'height': '10%', 'display': 'none'}),

        html.Br(),
        html.Label(['Choose Chart:'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='ddp_chart',
                     options=options_chart,

                     value='Avg Hourly Gas Levels for the UK',
                     disabled=False,
                     multi=False,
                     searchable=True,
                     search_value='',
                     placeholder='Please select...',
                     clearable=False,
                     style={'width': "100%"},
                     ),

        html.Label(['Choose Region:'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='ddp_region',
                     options=options_regions,
                     # height/space between dropdown options
                     value='',  # dropdown value selected automatically when page loads
                     disabled=False,  # disable dropdown value selection
                     multi=True,  # allow multiple dropdown values to be selected
                     searchable=True,  # allow user-searching of dropdown values
                     search_value='',  # remembers the value searched in dropdown
                     placeholder='All',  # gray, default text shown when no option is selected
                     clearable=True,  # allow user to removes the selected value
                     style={'width': "100%"},  # use dictionary to define CSS styles of your dropdown
                     ),

        html.Label(['Choose Settlement Type:'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='ddp_settlement',
                     options=options_settlements,

                     value='',
                     disabled=False,
                     multi=True,
                     searchable=True,
                     search_value='',
                     placeholder='All',
                     clearable=True,
                     style={'width': "100%"},
                     ),

        html.Label(['Choose Gas Type:'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='ddp_gas_type',
                     options=options_type,

                     value=['All'],
                     disabled=False,
                     multi=True,
                     searchable=True,
                     search_value='',
                     placeholder='All',
                     clearable=True,
                     style={'width': "100%"},
                     ),
        html.Label(['Choose Location:'], style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='ddp_loc',
                     options=options_loc,

                     value='',
                     disabled=False,
                     multi=True,
                     searchable=True,
                     search_value='',
                     placeholder='All',
                     clearable=True,
                     style={'width': "100%"},
                     ),
        html.Label(['Dates:'], style={'font-weight': 'bold', "text-align": "center"}),
        html.Br(),
        dcc.DatePickerRange(
            id='my-range-slider',
            end_date=date(2021, 5, 31),
            start_date=date(2021, 3, 1),
            display_format='Y MMMM DD',
            start_date_placeholder_text='MMMM Y, DD',
            min_date_allowed="2021-03-01",
            max_date_allowed="2021-05-31",
            initial_visible_month="2021-05-31",
            style=dict(font=dict(size=10))
        )

    ], className='three columns'),

])


# ---------------------------------------------------------------
# ---------------------------------------------------------------
@app.callback(
    Output(component_id='our_graph', component_property='figure'),
    # Output(component_id='ddp_region', component_property="options"),
    # Output(component_id='ddp_gas_type', component_property='options'),
    # Output(component_id='ddp_settlement', component_property='options'),
    Output(component_id='ddp_loc', component_property='options'),
    Output(component_id='img_box_plot', component_property='style'),
    [Input(component_id='ddp_chart', component_property='value')],
    [Input(component_id='ddp_region', component_property='value')],
    [Input(component_id='ddp_gas_type', component_property='value')],
    [Input(component_id='ddp_settlement', component_property='value')],
    [Input(component_id='ddp_loc', component_property='value')],
    [Input(component_id='my-range-slider', component_property='start_date')],
    [Input(component_id='my-range-slider', component_property='end_date')],
)
def build_graph(chart, region, gas_type, settlement, location, start_date, end_date):
    dff = check_data(region, gas_type, settlement, location, start_date, end_date)
    drop_downs = update_drop_downs(dff)
    if chart == 'Avg Hourly Gas Levels by Settlement':
        fig = build_bar_chart(dff, start_date, end_date)
        image_show = {'display': 'none'}
    elif chart == 'Avg Hourly Gas Levels by Region':
        fig = build_bar_chart_regions(dff, start_date, end_date)
        image_show = {'display': 'none'}
    elif chart == 'Avg Hourly Gas Levels by Week Day':
        fig = build_box_chart(dff, start_date, end_date)
        image_show = {'display': 'block'}
    elif chart == 'Avg Hourly Gas Levels by Hour':
        fig = build_line_chart(dff, start_date, end_date)
        image_show = {'display': 'none'}
    elif chart == 'Avg Hourly Gas Levels for the UK':
        fig = build_uk_chart(dff, start_date, end_date)
        image_show = {'display': 'none'}
    return (fig, drop_downs, image_show)


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def build_uk_chart(dff, start_date, end_date):
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, annotations=[
            {"text": "No Data Found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}])
    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(1, 50))
        min_lat, max_lat = 48.77, 60
        min_lon, max_lon = -9.05, 5
        dff = dff.groupby(['region', 'loc', 'settlement_types', 'Latitude', 'Longitude']).describe().Value.reset_index()
        lst_elements = sorted(list(dff['settlement_types'].unique()))
        dff["size"] = scaler.fit_transform(
            dff['mean'].values.reshape(-1, 1)).reshape(-1)
        dff["size"] = dff["size"].fillna(0)
        dff['mean'] = dff['mean'].fillna(0)
        fig = px.scatter_mapbox(dff,
                                color="settlement_types",  # which column to use to set the color of markers
                                hover_name="loc",  # column added to hover information
                                hover_data=['mean', 'max', 'min'],
                                size="size",  # size of markers
                                lat='Latitude',
                                lon='Longitude',
                                zoom=6,
                                mapbox_style="carto-positron",
                                center=dict(lat=52.4796992, lon=-1.9026911),
                                labels={"settlement_types": "Settlement Type", "mean": "Avg Hourly Gas Levels (ug/m3)"}

                                )
        fig.update_layout(title={'text': 'Avg Hourly Gas Levels for the UK ({} - {})'.format(start_date, end_date),
                                 'font': {'size': 14}, 'x': 0.5, 'xanchor': 'center'}, showlegend=True,
                          legend={'itemsizing': 'trace'})
        fig.add_layout_image(
            dict(
                source=app.get_asset_url('436.jpg'),
                xref="paper", yref="paper",
                # x=1, y=1.05,
                sizex=0.2, sizey=0.2,
                xanchor="left", yanchor="bottom"
            )
        )
        fig.update_layout(legend={'itemsizing': 'constant'})
    return fig


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def build_bar_chart(dff, start_date, end_date):
    if dff.empty:
        fig = px.bar()
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, annotations=[
            {"text": "No Data Found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}])
    else:

        dff = dff.groupby(['settlement_types', 'Type']).describe().Value.reset_index()
        dff['Int'] = np.around(dff['mean']).astype(int)
        fig = px.bar(dff, x='settlement_types', y='mean', text='Int', color="Type",
                     labels={"settlement_types": "Settlement Type", "Type": "Gas Type",
                             "mean": "Avg Hourly Gas Levels (ug/m3)"})
        # fig.update_traces(textinfo='percent+label')
    fig.update_layout(title={'text': 'Avg Hourly Gas Levels by Settlement ({} - {})'.format(start_date, end_date),
                             'font': {'size': 14}, 'x': 0.5, 'xanchor': 'center'}, barmode='group')
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def build_bar_chart_regions(dff, start_date, end_date):
    if dff.empty:
        fig = px.bar()
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, annotations=[
            {"text": "No Data Found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}])
    else:
        dff = dff.groupby(['region', 'Type']).describe().Value.reset_index()
        dff['Int'] = np.around(dff['mean']).astype(int)
        fig = px.bar(dff, x='region', y='mean', text='Int', color="Type",
                     labels={"Int": "Rounded", "region": "Region", "Type": "Gas Type",
                             "mean": "Avg Hourly Gas Levels (ug/m3)"})
        # fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        title={'text': 'Avg Hourly Gas Levels by Region ({} - {})'.format(start_date, end_date), 'font': {'size': 14},
               'x': 0.5, 'xanchor': 'center'}, barmode='group')
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def build_box_chart(dff, start_date, end_date):
    if dff.empty:
        fig = px.box()
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, annotations=[
            {"text": "No Data Found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 28}}])
    else:
        # dff = dff.groupby(['Date','weekday_no','Type']).describe().Value.reset_index()
        # dff = dff.sort_values(by=['weekday_no'])
        # dff['weekday'] = dff['weekday_no'].map(dayOfWeek)
        # fig = px.box(dff,x='weekday',y='mean',color="Type", labels={
        # "weekday": "Week Day", "mean": "Avg Hourly Gas Levels (ug/m3)"} )
        # fig.update_traces(textinfo='percent+label')
        settlement_types_to_show = sorted(list(dff['settlement_types'].astype(str).unique()))
        gas_types_to_show = sorted(list(dff['Type'].astype(str).unique()))
        fig = make_subplots(rows=len(settlement_types_to_show), cols=1, subplot_titles=(settlement_types_to_show))
        dff = dff.groupby(['settlement_types', 'Date', 'weekday_no', 'Type']).describe().Value.reset_index()
        dff = dff.sort_values(by=['weekday_no'])
        dff['weekday'] = dff['weekday_no'].map(dayOfWeek)
        graph_no = 1
        for settlement_types in settlement_types_to_show:
            dff_sub = dff[dff['settlement_types'] == settlement_types]
            col_num = 0
            for gas_type in gas_types_to_show:
                dff_sub_gas = dff_sub[dff_sub['Type'] == gas_type]
                if graph_no == 1:
                    fig.append_trace(
                        go.Box(x=dff_sub_gas['weekday'], y=dff_sub_gas['mean'], offsetgroup=str(col_num), name=gas_type,
                               line=dict(color='black'), legendgroup=col_num, fillcolor=COLORS[col_num],
                               marker=dict(outliercolor="red"), alignmentgroup=str(graph_no)), row=graph_no, col=1)
                else:
                    fig.append_trace(
                        go.Box(x=dff_sub_gas['weekday'], y=dff_sub_gas['mean'], offsetgroup=str(col_num), name=gas_type,
                               line=dict(color='black'), legendgroup=col_num, fillcolor=COLORS[col_num],
                               marker=dict(outliercolor="red"), showlegend=False, ), row=graph_no, col=1)
                col_num += 1

            if graph_no == 1:
                fig['layout']['xaxis']['title'] = 'Week Day'
                fig['layout']['yaxis']['title'] = 'Avg Hourly Gas Levels (ug/m3)'

            else:
                fig['layout']['xaxis' + str(graph_no)]['title'] = 'Week Day'
                fig['layout']['yaxis' + str(graph_no)]['title'] = 'Avg Hourly Gas Levels (ug/m3)'

            graph_no += 1
    fig.update_layout(boxmode='group', boxgap=0.25, boxgroupgap=0.25)
    fig.update_layout(
        title={'text': 'Avg Hourly Distribution Gas Levels by Week Day ({} - {})'.format(start_date, end_date),
               'font': {'size': 14}, 'x': 0.5, 'xanchor': 'center'})
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------
# ---------------------------------------------------------------
def build_line_chart(dff, start_date, end_date):
    if dff.empty:
        fig = px.line()
        fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, annotations=[
            {"text": "No Data Found", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14}}])
    else:
        dff = dff.groupby(['Time', 'weekday_no', 'Type']).describe().Value.reset_index()
        gas_types_to_show = sorted(list(dff['Type'].astype(str).unique()))
        fig = make_subplots(rows=len(gas_types_to_show), cols=1, subplot_titles=(gas_types_to_show))
        graph_no = 1
        for gas_type_plot in gas_types_to_show:
            dff_sub = dff[dff['Type'] == gas_type_plot]
            dff_sub = dff_sub.pivot(index=["Time"], columns=['weekday_no'], values='mean')
            dff_sub.columns = dff_sub.columns.map(dayOfWeek)
            col_no = 0
            for col in dff_sub.columns:
                if graph_no == 1:

                    fig.append_trace(go.Scatter(x=dff_sub.index,
                                                y=dff_sub[col].values,
                                                line=dict(color=COLORS[col_no]),
                                                name=col,
                                                legendgroup=col,
                                                mode='lines'
                                                ), row=graph_no, col=1)
                else:
                    fig.append_trace(go.Scatter(x=dff_sub.index,
                                                y=dff_sub[col].values,
                                                line=dict(color=COLORS[col_no]),
                                                name=col,
                                                legendgroup=col,
                                                mode='lines',
                                                showlegend=False), row=graph_no, col=1)
                col_no += 1
            if graph_no != 1:
                fig['layout']['xaxis' + str(graph_no)]['title'] = 'Hour'
                fig['layout']['yaxis' + str(graph_no)]['title'] = 'Avg Hourly Gas Levels (ug/m3)'
            graph_no += 1
# fig = px.line(dff,x='Time',y='mean',color='weekday',
    # labels={ "weekday": "Week Day", "mean": "Avg Gas Level (ug/m3)","Time":"Hour"},
    # range_y=[0,dff['max']],range_x=['00:00:00','24:00:00'] )
    fig.update_layout(
        title={'text': 'Avg Hourly Gas Levels by Hour (March 2021 - May 2021)', 'font': {'size': 14}, 'x': 0.5,
               'xanchor': 'center'},
        xaxis_title="Hour",
        yaxis_title="Avg Hourly Gas Levels (ug/m3)",
        legend_title="Gas Types",
        font=dict(
            size=10
        )
        )
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---------------------------------------------------------------

def update_drop_downs(dff):
    options_loc = []
    for location in sorted(list(dff['loc'].astype(str).unique())):
        options_loc.append({'label': location, 'value': location})

    dropdown_results = options_loc
    return (dropdown_results)


# ---------------------------------------------------------------
def check_data(region, gas_type, settlement, location, start_date, end_date):
    dff = df[(df["settlement_types"].isnull() != True)]
    dff = dff[(dff['Date'] >= start_date) & (dff['Date'] <= end_date)]
    if settlement != "All" and bool(settlement):
        dff = dff[(dff["settlement_types"].isin(settlement))]
    if region != "All" and bool(region):
        dff = dff[(dff["region"].isin(region))]
    if gas_type != "All" and bool(gas_type):
        dff = dff[(dff["Type"].isin(gas_type))]
    if location != "All" and bool(location):
        dff = dff[(dff["loc"].isin(location))]
    return (dff)


# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)
    print("Hi")


