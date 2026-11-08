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
from dash.exceptions import PreventUpdate



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
def plot_series(pasado, pron, real, fecha):
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
    # Alertas en horas de déficit
    for hora in pron.index[pron["alerta"]]:
        fig.add_vrect(x0=hora - pd.Timedelta(minutes=30), x1=hora + pd.Timedelta(minutes=30),
                      fillcolor="#ff6b6b", opacity=0.15, line_width=0, layer="below")
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

def estilo_oscuro(fig):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color="#2cfec1", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor='#7C7C7C')
    return fig


def figura_mezcla(data):
    # Agrupar las fuentes
    grupos = pd.DataFrame({
        "Gas": data["gas"],
        "Carbón": data["coal"],
        "Fuel/diésel": data["liquid_fuel"],
        "Importación": data[["india_bheramara_hvdc", "india_tripura", "india_adani", "nepal"]].sum(axis=1),
        "Renovables": data[["hydro", "solar", "wind"]].sum(axis=1),
    })
    # Totales generación
    anual = grupos.groupby(data.index.year).sum()
    pct = anual.div(anual.sum(axis=1), axis=0) * 100

    colores = {"Gas": "#2a78d6", "Carbón": "#eb6834", "Fuel/diésel": "#ffb347",
               "Importación": "#7fafdf", "Renovables": "#2cfec1"}
    fig = go.Figure([go.Bar(name=c, x=pct.index, y=pct[c], marker_color=colores[c])
                     for c in pct.columns])
    fig.update_layout(barmode="stack", yaxis_title="% de la generación")
    return estilo_oscuro(fig)


def figura_racionamiento(data):
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    # Tabla de racionamiento promedio
    tabla = data.pivot_table(index=data.index.hour, columns=data.index.month,
                             values="load_shedding", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=tabla.values,
        x=[meses[m - 1] for m in tabla.columns],
        y=tabla.index,
        colorscale="Reds",
        colorbar=dict(title="MW"),
        hovertemplate="%{x}, %{y}:00 h<br>%{z:.0f} MW en promedio<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Hora del día", dtick=3, showgrid=False)
    return estilo_oscuro(fig)

def pronosticar(historia):
    futuro = pd.date_range(historia.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    serie = historia.reindex(historia.index.append(futuro))
    X = F.construir(serie)
    pred = modelo.predict(X.loc[futuro])
    gen_disp = serie["generation_mw"].shift(24).loc[futuro]
    res = pd.DataFrame({"pronostico": pred, "gen_disponible": gen_disp.values}, index=futuro)
    res["deficit"] = res["pronostico"] - res["gen_disponible"]
    res["alerta"] = res["deficit"] > 0
    return res

def mw(x):
    # 12345.6 -> "12.346"
    return f"{x:,.0f}".replace(",", ".")


def accion_sugerida(pron):
    horas = int(pron["alerta"].sum())
    if horas == 0:
        return "Demanda cubierta"
    if horas <= 3 and pron["deficit"].max() < 1000:
        return "Déficit corto (hasta 3 h y menor a 1.000 MW)"
    return ("Déficit prolongado (más de 3 h) o grande (1.000 MW o más): activar la reserva")


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
                    #Dropdown Hora
                    html.Div(
                        id="componente-hora",
                        children=[
                            dcc.Dropdown(
                                id="dropdown-hora-inicial-hora",
                                options=[{"label": i, "value": i} for i in np.arange(0,24)],
                                value=pd.to_datetime(max(data.index)-dt.timedelta(days=7)).hour,
                                clearable=False,
                                # style=dict(width='50%', display="inline-block")
                            )
                        ],
                        style=dict(width='30%')
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
                        value=48,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ]
            )     
     
        ]
    )

def tarjeta(titulo, id_valor):
    return html.Div(
        className="kpi",
        children=[
            html.P(titulo, className="kpi-titulo"),
            html.H4("—", id=id_valor, className="kpi-valor"),
        ],
    )


def alert_card():
    return html.Div(
        id="alert-card",
        children=[
            html.P("Horas en alerta:"),
            html.Ul(id="lista-alertas", style={"maxHeight": "220px", "overflowY": "auto"}),
            html.P("Acción sugerida:"),
            html.Div(id="accion"),
        ],
    )

app.layout = html.Div(
    id="app-container",
    children=[
        
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card(), alert_card()]
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

                html.Div(
                    id="kpis",
                    children=[
                        tarjeta("Demanda pico esperada", "kpi-pico"),
                        tarjeta("Horas en alerta", "kpi-horas"),
                        tarjeta("Déficit máximo esperado", "kpi-deficit"),
                        tarjeta("Error del modelo (MAPE)", "kpi-mape"),
                    ],
                ),


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
                #Paneles Históricos            
                html.Div(
                    id="historicos",
                    children=[
                        html.Div(
                            className="panel-historico",
                            children=[
                                html.B("Mezcla de generación por fuente"),
                                html.Hr(),
                                dcc.Graph(figure=figura_mezcla(data)),
                            ],
                        ),
                        html.Div(
                            className="panel-historico",
                            children=[
                                html.B("Racionamiento histórico por hora y mes"),
                                html.Hr(),
                                dcc.Graph(figure=figura_racionamiento(data)),
                            ],
                        ),
                    ],
                ),

            
            ],
        ),
    ],
)


@app.callback(
    [Output(component_id="plot_series", component_property="figure"),
     Output(component_id="kpi-pico", component_property="children"),
     Output(component_id="kpi-horas", component_property="children"),
     Output(component_id="kpi-deficit", component_property="children"),
     Output(component_id="kpi-mape", component_property="children"),
     Output(component_id="lista-alertas", component_property="children"),
     Output(component_id="accion", component_property="children")],
    [Input(component_id="datepicker-inicial", component_property="date"),
     Input(component_id="dropdown-hora-inicial-hora", component_property="value"),
     Input(component_id="slider-proyeccion", component_property="value")]
)
def update_output_div(date, hour, proy):
    if date is None or hour is None or proy is None:
        raise PreventUpdate

    fecha = pd.to_datetime(date + " " + str(hour) + ":00", format="%Y-%m-%d %H:%M")

    historia = data.loc[:fecha].tail(24 * 14)
    pron = pronosticar(historia)
    real = data["demand_mw"].reindex(pron.index)
    pasado = historia.tail(int(proy))
    alertas = pron[pron["alerta"]]

    pico = mw(pron["pronostico"].max()) + " MW"
    horas = "1 hora" if len(alertas) == 1 else str(len(alertas)) + " horas"
    deficit = "−" + mw(alertas["deficit"].max()) + " MW" if len(alertas) > 0 else "Sin déficit"
    if real.notna().all():
        mape = ((real - pron["pronostico"]).abs() / real).mean() * 100
        texto_mape = [f"{mape:.1f} %".replace(".", ","), html.Span("este día", className="kpi-nota")]
    else:
        texto_mape = ["4,9 %", html.Span("año de prueba", className="kpi-nota")]

    lista = [html.Li(f"{hora:%d/%m %H:%M} · −{mw(d)} MW") for hora, d in alertas["deficit"].items()]
    if len(lista) == 0:
        lista = [html.Li("Ninguna hora en alerta")]

    figura = plot_series(pasado, pron, real, fecha)
    return figura, pico, horas, deficit, texto_mape, lista, accion_sugerida(pron)

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
