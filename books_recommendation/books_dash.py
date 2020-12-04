import dash
import dash_html_components as html
import dash_core_components as dcc
import glob
import texttograph as t2g
import pandas as pd 
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
from loguru import logger

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server

def get_options():
    """ Lista libros"""
    #BOOKS_PATH='../books/*epub'

    #ebooks= sorted(glob.glob(BOOKS_PATH) )
    options = []
    ebooks = pd.read_csv("vectorized.csv")['title'].values
    for title in ebooks:
        logger.debug(title)
        #title=ebook.replace('../books/','').replace('.epub','')
        options.append({ "label" : title ,   'value': title})
    return options

def prepare_graphx(busqueda):
    df_distance = pd.read_csv("distances.csv",index_col='title')
    df_results,G = t2g.get_chart(busqueda,
                df_distance ,
                number_elements_primary=8,
                number_elements_secondary=3)
    network = t2g.Display_Interactive_Chart(G)            
    return fig
    
fig = px.scatter()
options = get_options()
G = nx.Graph( )

app.layout = html.Div(
        [
        dcc.Dropdown(
            id='dropdown',
            options= options    ),
        html.Iframe(src="index2.html",
                style={"height": "1067px", "width": "100%"}),
        html.Div(className='row', children=[
        html.Div([html.H2('Overall Data'),
                    html.P('Num of nodes: ' + str(len(G.nodes))),
                    html.P('Num of edges: ' + str(len(G.edges)))],
                    className='three columns'),
        html.Div([
                html.H2('Selected Data'),
                html.Div(id='selected-data'),
            ], className='six columns')
        ])
        ]
    )


@app.callback(
    dash.dependencies.Output('Graph', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_output(value):
    busqueda = value
    return prepare_graphx(busqueda)


if __name__ == '__main__':
    app.run_server(debug=True)