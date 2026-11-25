"""Tablero de pronóstico de demanda eléctrica y alertas de racionamiento (PGCB, Bangladesh).

Enviamos a la API la historia reciente y grafica lo que devuelve.

    python -m tablero.app          en la raíz del repo


"""
import datetime as dt
import os

import dash
import pandas as pd
import plotly.graph_objs as go
import requests
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

API_URL = os.getenv("API_URL", "http://localhost:8001").rstrip("/")
PORT = int(os.getenv("PORT", "8050"))
RUTA_DATOS = os.getenv("PGCB_DATOS", "data/processed/pgcb_horario_limpio.csv")

COLUMNAS_API = ["demand_mw", "generation_mw", "load_shedding", "liquid_fuel"]
HISTORIA_H = 14 * 24          # Envío de 2 semanas
UMBRAL_HORAS = 3              # Número de horas para activar reserva
UMBRAL_PROB_ALTA = 0.80       # Activación de reserva por probabilidad


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Pronóstico demanda PGCB"
server = app.server
app.config.suppress_callback_exceptions = True


# ---------------------------------------------------------------- data
def load_data():
    df = pd.read_csv(RUTA_DATOS, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


data = load_data()


# ---------------------------------------------------------------- comunicación con la API
class ErrorAPI(Exception):
    """La API respondió con error, revisar vacíos o una historia menor a la requerida(14 días)"""


def pronosticar(historia):
    h = historia[COLUMNAS_API].tail(HISTORIA_H).reset_index()
    h["datetime"] = h["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    h = h.astype(object).where(h.notna(), None)  # NaN -> null en el JSON
    r = requests.post(f"{API_URL}/api/v1/predict", json={"historia": h.to_dict("records")}, timeout=30)
    if r.status_code != 200:
        try:
            detalle = r.json().get("detail")
        except ValueError:
            detalle = r.text
        raise ErrorAPI(f"la API respondió {r.status_code}: {detalle}")

    cuerpo = r.json()
    pron = pd.DataFrame(cuerpo["predicciones"])
    pron["datetime"] = pd.to_datetime(pron["datetime"])
    pron = pron.set_index("datetime")
    numericas = ["demanda_mw", "generacion_disponible_mw", "deficit_mw", "prob_racionamiento"]
    pron[numericas] = pron[numericas].astype(float)  # null -> NaN
    pron["alerta"] = pron["alerta"].astype(bool)
    return pron, cuerpo


def calcular(date, hour):
    fecha = pd.Timestamp(str(date)[:10]) + pd.Timedelta(hours=int(hour))
    historia = data.loc[:fecha].tail(HISTORIA_H)
    pron, cuerpo = pronosticar(historia)
    real = data["demand_mw"].reindex(pron.index)
    return historia, pron, real, cuerpo


# ---------------------------------------------------------------- formato
def mw(x):
    # 12345.6 -> "12.346"
    return f"{x:,.0f}".replace(",", ".")


def pct(p):
    return f"{p * 100:.0f} %"


def texto_deficit(x):
    if pd.isna(x):
        return "sin dato de generación"
    return f"déficit {mw(x)} MW" if x > 0 else f"holgura {mw(-x)} MW"


def accion_sugerida(pron):
    horas = int(pron["alerta"].sum())
    pmax = float(pron["prob_racionamiento"].max())
    if horas == 0:
        return "Demanda cubierta: ninguna hora supera el umbral de alerta."
    if horas > UMBRAL_HORAS or pmax >= UMBRAL_PROB_ALTA:
        return (f"Activar la reserva: {horas} h en alerta y probabilidad máxima de {pct(pmax)} "
                f"(criterio: más de {UMBRAL_HORAS} h o probabilidad de {pct(UMBRAL_PROB_ALTA)} o más).")
    return (f"Vigilar: {horas} h en alerta con probabilidad máxima de {pct(pmax)}; "
            "tener la reserva lista por si el riesgo aumenta.")


# ---------------------------------------------------------------- figuras
def plot_series(pasado, pron, real, fecha, umbral):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.06)
    prob = pron["prob_racionamiento"] * 100

    # Panel 1: demanda y generación
    fig.add_trace(go.Scatter(name="Demanda (historia)", x=pasado.index, y=pasado["demand_mw"],
                             mode="lines", line=dict(color="#2cfec1", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(name="Pronóstico", x=pron.index, y=pron["demanda_mw"],
                             mode="lines+markers", line=dict(color="#ffb347", width=3),
                             customdata=prob,
                             hovertemplate="%{y:,.0f} MW<br>Prob. racionamiento: %{customdata:.0f} %"
                                           "<extra>Pronóstico</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(name="Demanda real", x=real.index, y=real,
                             mode="lines", line=dict(color="#2cfec1", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(name="Generación disponible", x=pron.index, y=pron["generacion_disponible_mw"],
                             mode="lines", line=dict(color="#7fafdf", dash="dash")), row=1, col=1)

    # Panel 2: probabilidad de racionamiento y umbral del modelo
    fig.add_trace(go.Bar(name="Prob. racionamiento", x=pron.index, y=prob, showlegend=False,
                         marker_color=["#ff6b6b" if a else "#7fafdf" for a in pron["alerta"]],
                         width=3600 * 1000 * 0.8,
                         hovertemplate="%{y:.0f} %<extra>Prob. racionamiento</extra>"), row=2, col=1)
    fig.add_hline(y=umbral * 100, line_dash="dash", line_color="#ff6b6b", row=2, col=1,
                  annotation_text=f"umbral {pct(umbral)}", annotation_position="top left",
                  annotation_font_color="#ff6b6b")

    # Horas en alerta sombreadas en ambos paneles y línea del momento elegido
    for hora in pron.index[pron["alerta"]]:
        fig.add_vrect(x0=hora - pd.Timedelta(minutes=30), x1=hora + pd.Timedelta(minutes=30),
                      fillcolor="#ff6b6b", opacity=0.15, line_width=0, layer="below", row="all", col=1)
    fig.add_vline(x=fecha, line_dash="dot", line_color="#7C7C7C", row="all", col=1)

    fig.update_layout(height=540, hovermode="x", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#2cfec1")
    fig.update_xaxes(showgrid=True, gridwidth=0.25, gridcolor="#7C7C7C")
    fig.update_yaxes(showgrid=True, gridwidth=0.25, gridcolor="#7C7C7C")
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="Prob. (%)", range=[0, 100], row=2, col=1)
    return fig


def figura_vacia(mensaje):
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                       font=dict(color="#ff6b6b", size=14))
    fig.update_layout(height=540, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
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
    pct_anual = anual.div(anual.sum(axis=1), axis=0) * 100

    colores = {"Gas": "#2a78d6", "Carbón": "#eb6834", "Fuel/diésel": "#ffb347",
               "Importación": "#7fafdf", "Renovables": "#2cfec1"}
    fig = go.Figure([go.Bar(name=c, x=pct_anual.index, y=pct_anual[c], marker_color=colores[c])
                     for c in pct_anual.columns])
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


# ---------------------------------------------------------------- componentes
def description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H3("Pronóstico de demanda eléctrica · PGCB"),
            html.Div(
                id="intro",
                children="Elija una fecha y hora: el tablero envía a la API las dos semanas anteriores "
                         "y recibe el pronóstico de demanda de las 24 horas siguientes, la generación "
                         "disponible estimada y la probabilidad de racionamiento de cada hora. Si la "
                         "fecha ya pasó, también muestra lo que realmente ocurrió.",
            ),
        ],
    )


def generate_control_card():
    return html.Div(
        id="control-card",
        children=[
            html.P("Seleccionar fecha y hora inicial:"),
            html.Div(
                id="componentes-fecha-inicial",
                children=[
                    html.Div(
                        id="componente-fecha",
                        children=[
                            dcc.DatePickerSingle(
                                id="datepicker-inicial",
                                min_date_allowed=dt.date(2024, 6, 18),
                                max_date_allowed=max(data.index.date),
                                initial_visible_month=max(data.index.date),
                                date=max(data.index.date),
                            )
                        ],
                        style=dict(width="40%"),
                    ),
                    html.P(" ", style=dict(width="5%", textAlign="center")),
                    html.Div(
                        id="componente-hora",
                        children=[
                            dcc.Dropdown(
                                id="dropdown-hora-inicial-hora",
                                options=[{"label": f"{i:02d}:00", "value": i} for i in range(24)],
                                value=int(max(data.index).hour),
                                clearable=False,
                            )
                        ],
                        style=dict(width="30%"),
                    ),
                ],
                style=dict(display="flex"),
            ),
            html.Br(),
            html.Div(
                id="campo-slider",
                children=[
                    html.P("Días de historia a mostrar:"),
                    dcc.Slider(id="slider-proyeccion", min=1, max=7, step=1, value=2,
                               marks={i: f"{i} d" for i in range(1, 8)}),
                ],
            ),
        ],
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
        # Columna izquierda
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card(), alert_card(),
                      html.P(id="estado-api", className="kpi-nota")],
        ),
        # Columna derecha
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                html.Div(
                    id="kpis",
                    children=[
                        tarjeta("Demanda pico esperada", "kpi-pico"),
                        tarjeta("Horas en alerta", "kpi-horas"),
                        tarjeta("Probabilidad máxima de racionamiento", "kpi-prob"),
                        tarjeta("Error del modelo (MAPE)", "kpi-mape"),
                    ],
                ),
                html.Div(
                    id="model_graph",
                    children=[
                        html.B("Demanda y generación [MW] · probabilidad de racionamiento [%]"),
                        html.Hr(),
                        dcc.Graph(id="plot_series"),
                    ],
                ),
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


# ---------------------------------------------------------------- interacción
@app.callback(
    [Output("plot_series", "figure"),
     Output("kpi-pico", "children"),
     Output("kpi-horas", "children"),
     Output("kpi-prob", "children"),
     Output("kpi-mape", "children"),
     Output("lista-alertas", "children"),
     Output("accion", "children"),
     Output("estado-api", "children")],
    [Input("datepicker-inicial", "date"),
     Input("dropdown-hora-inicial-hora", "value"),
     Input("slider-proyeccion", "value")],
)
def update_output_div(date, hour, proy):
    if date is None or hour is None or proy is None:
        raise PreventUpdate

    try:
        historia, pron, real, cuerpo = calcular(date, hour)
    except (requests.RequestException, ErrorAPI) as e:
        return (figura_vacia("No se pudo obtener el pronóstico de la API"), "—", "—", "—", "—",
                [html.Li("Sin pronóstico")], f"No se pudo obtener el pronóstico: {e}",
                f"API sin conexión ({API_URL})")

    fecha = historia.index[-1]
    pasado = historia.tail(int(proy) * 24)
    alertas = pron[pron["alerta"]]
    resumen = cuerpo["resumen"]
    umbral = float(cuerpo["umbral_alerta"])

    hora_pico = pd.Timestamp(resumen["hora_pico"])
    pico = [mw(resumen["demanda_pico_mw"]) + " MW",
            html.Span(f"{hora_pico:%d/%m} a las {hora_pico:%H:%M}", className="kpi-nota")]
    n = int(resumen["horas_en_alerta"])
    horas = "1 hora" if n == 1 else f"{n} horas"
    prob_max = [pct(float(pron["prob_racionamiento"].max())),
                html.Span(f"umbral de alerta {pct(umbral)}", className="kpi-nota")]

    if real.notna().all():
        mape = ((real - pron["demanda_mw"]).abs() / real).mean() * 100
        texto_mape = [f"{mape:.1f} %".replace(".", ","), html.Span("este día", className="kpi-nota")]
    else:
        texto_mape = ["4,9 %", html.Span("año de prueba", className="kpi-nota")]

    lista = [html.Li(f"{hora:%d/%m %H:%M} · {pct(f.prob_racionamiento)} · {texto_deficit(f.deficit_mw)}")
             for hora, f in alertas.iterrows()]
    if not lista:
        lista = [html.Li("Ninguna hora en alerta")]

    estado = f"Modelo v{cuerpo['version_modelo']} · umbral {pct(umbral)} · API conectada"
    figura = plot_series(pasado, pron, real, fecha, umbral)
    return figura, pico, horas, prob_max, texto_mape, lista, accion_sugerida(pron), estado


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)