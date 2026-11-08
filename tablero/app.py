import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import joblib
from src import features as F
import datetime as dt



app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Pronóstico demanda PGCB"

server = app.server
app.config.suppress_callback_exceptions = True


# Load data from csv
def load_data():
    df = pd.read_csv('data/processed/pgcb_horario_limpio.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df
    

# Cargar datos
data = load_data()
modelo = joblib.load('models/modelo_demanda.joblib')

# Graficar serie
def plot_series(data, fecha, horas):
    historia = data.loc[:fecha].tail(24 * 14)
    pron = pronosticar(historia)
    real = data["demand_mw"].reindex(pron.index)
    pasado = historia.tail(horas)

    fig = go.Figure([
        go.Scatter(name='Demanda (historia)', x=pasado.index, y=pasado['demand_mw'],
                   mode='lines', line=dict(color="#2cfec1", width=2)),
        go.Scatter(name='Pronóstico', x=pron.index, y=pron['pronostico'],
                   mode='lines+markers', line=dict(color="#ffb347", width=3)),
        go.Scatter(name='Demanda real', x=real.index, y=real,
                   mode='lines', line=dict(color="#2cfec1", dash="dot")),
        go.Scatter(name='Generación disponible', x=pron.index, y=pron['gen_disponible'],
                   mode='lines', line=dict(color="#7fafdf", dash="dash")),
    ])
    fig.add_vline(x=fecha, line_dash="dot", line_color="#7C7C7C")

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title='MW',
        hovermode="x"
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#2cfec1")
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    return fig

def pronosticar(historia):
    futuro = pd.date_range(historia.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    serie = historia.reindex(historia.index.append(futuro))
    X = F.construir(serie)
    pred = modelo.predict(X.loc[futuro])
    gen_disp = serie["generation_mw"].shift(24).loc[futuro]
    return pd.DataFrame({"pronostico": pred, "gen_disponible": gen_disp.values}, index=futuro)   


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            #html.H5("Proyecto 1"),
            html.H3("Pronóstico de demanda eléctrica · PGCB"),
            html.Div(
                id="intro",
                children="Elija una fecha y hora: el tablero toma la historia hasta ese momento y "
                        "pronostica la demanda de las 24 horas siguientes, comparándola con la "
                        "generación disponible y con lo que realmente ocurrió."
            ),
        ],
    )


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[

            # Fecha inicial
            html.P("Seleccionar fecha y hora inicial:"),

            html.Div(
                id="componentes-fecha-inicial",
                children=[
                    html.Div(
                        id="componente-fecha",
                        children=[
                            dcc.DatePickerSingle(
                                id='datepicker-inicial',
                                min_date_allowed=dt.date(2024, 6, 18),
                                max_date_allowed=max(data.index.date),
                                initial_visible_month=max(data.index.date),
                                date=max(data.index.date)-dt.timedelta(days=7)
                            )
                        ],
                        style=dict(width='30%')
                    ),
                    
                    html.P(" ",style=dict(width='5%', textAlign='center')),
                    
                    html.Div(
                        id="componente-hora",
                        children=[
                            dcc.Dropdown(
                                id="dropdown-hora-inicial-hora",
                                options=[{"label": i, "value": i} for i in np.arange(0,24)],
                                value=pd.to_datetime(max(data.index)-dt.timedelta(days=7)).hour,
                                # style=dict(width='50%', display="inline-block")
                            )
                        ],
                        style=dict(width='20%')
                    ),
                ],
                style=dict(display='flex')
            ),

            html.Br(),

            # Slider proyección
            html.Div(
                id="campo-slider",
                children=[
                    html.P("Horas de historia a mostrar:"),
                    dcc.Slider(
                        id="slider-proyeccion",
                        min=24,
                        max=336,
                        step=24,
                        value=168,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ]
            )     
     
        ]
    )


app.layout = html.Div(
    id="app-container",
    children=[
        
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[


                # Grafica de la serie de tiempo
                html.Div(
                    id="model_graph",
                    children=[
                        html.B("Demanda y generación horaria [MW]"),
                        html.Hr(),
                        dcc.Graph(
                            id="plot_series",  
                        )
                    ],
                ),

            
            ],
        ),
    ],
)


@app.callback(
    Output(component_id="plot_series", component_property="figure"),
    [Input(component_id="datepicker-inicial", component_property="date"),
    Input(component_id="dropdown-hora-inicial-hora", component_property="value"),
    Input(component_id="slider-proyeccion", component_property="value")]
)
def update_output_div(date, hour, proy):

    if ((date is not None) & (hour is not None) & (proy is not None)):
        hour = str(hour)
        minute = str(0)

        initial_date = date + " " + hour + ":" + minute
        initial_date = pd.to_datetime(initial_date, format="%Y-%m-%d %H:%M")

        # Graficar
        plot = plot_series(data, initial_date, int(proy))
        return plot


# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
