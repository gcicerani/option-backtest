# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:21:01 2025

@author: GCicerani
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 18:17:32 2025

@author: GCicerani
"""



import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, dash_table
import pandas as pd
import plotly.express as px
from MosaicSurfacef import Surface
import matplotlib.pyplot as plt
import base64
import io
from analysis_script import run_analysis


# external_stylesheets=[dbc.themes.CYBORG]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
# app = dash.Dash(__name__, suppress_callback_exceptions=True)

# app.layout = html.Div([
#     html.H2("Option Backtest"),
    
#     html.Label("Choose a start date:"),
#     dcc.DatePickerSingle(
#         id='data-osservazione',
#         date=None
#     ),
    
#     html.Label("choose a end date:"),
#     dcc.DatePickerSingle(
#         id='end-date',
#         date=None,  # valore di default
#     ),
    
    
#     html.Button("Load Data", id='carica-dati', n_clicks=0),
    
#     html.Br(), html.Hr(),
    
#     html.Div(id='tabella-output'),
#     html.Div(id='input-manuale'),
    
#     html.Br(), html.Hr(),
    
#     # Bottone genera grafici
#     html.Button("Backtest", id="btn-grafici", n_clicks=0),
    
#     html.Div(id="output-grafici", style={
#         "marginTop": "20px", "maxHeight": "600px", "overflowY": "auto",
#         "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"
#     }),
#     dcc.Store(id='store-expiries') 
    
# ])

app.layout = dbc.Container([
    # Immagine intestazione (assicurati che sia salvata in /assets/)
    # html.Img(src='/assets/ImmagineDash.png', style={"width": "100%", "marginBottom": "30px"}),

    html.H2("📈 Option Backtest", className="text-center mb-4"),

    # RIGA DATE & LOAD DATA
    dbc.Row([
        dbc.Col([
            html.Label("Choose a start date:"),
            dcc.DatePickerSingle(id='data-osservazione', date=None),
        ], width=3),

        dbc.Col([
            html.Label("Choose an end date:"),
            dcc.DatePickerSingle(id='end-date', date=None),
        ], width=3),

        dbc.Col([
            html.Label(" "),
            dbc.Button("📂 Load Data", id='carica-dati', n_clicks=0, color="primary", className="mt-2"),
        ], width=2),
    ], className="mb-4"),

    html.Hr(),

    # TABELLE
    dbc.Row([
        dbc.Col([
            html.Div(id='tabella-output')
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Div(id='input-manuale')
        ])
    ], className="mb-4"),

    html.Hr(),

    # Bottone backtest
    dbc.Row([
        dbc.Col([
            dbc.Button("📊 Run Backtest", id="btn-grafici", n_clicks=0, color="success", className="mb-3")
        ])
    ]),

    # Output grafici
    html.Div(id="output-grafici", style={
        "marginTop": "20px",
        "maxHeight": "600px",
        "overflowY": "auto",
        "display": "grid",
        "gridTemplateColumns": "repeat(4, 1fr)",
        "gap": "10px"
    }),

    dcc.Store(id='store-expiries')
], fluid=True)



@app.callback(
    Output('tabella-output', 'children'),
    # Output('parametri-dropdown', 'options'),
    Output('input-manuale', 'children'),
    Output('store-expiries', 'data'),
    Input('carica-dati', 'n_clicks'),
    State('data-osservazione', 'date')
)
def carica_dati(n, data_sel):
    if not data_sel:
        return "Choose a start date first", [], []
    
    # LANCIA SCRIPT BACKEND CON LA DATA
    Output = Surface(data_sel)
    df = Output[0]  # <-- tuo script
    dfBE = Output[1]
    ExpireTable = Output[2][0:12]
    
    
    df_display = df.copy()
    df_display["index"] = df_display.index
    df_display = df_display[["index"] + [col for col in df_display.columns if col != "index"]]
    
    # Crea tabella
    table = dash.dash_table.DataTable(
        data=[
       {str(k): v for k, v in row.items()}
       for row in df_display.to_dict("records")
   ],
   columns=[
       {"name": str(col), "id": str(col)}
       for col in df_display.columns
   ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        style_data_conditional=[
    {
        'if': {'column_id': 'index'},
        'backgroundColor': '#d6e9f9',  # azzurrino chiaro
        'color': 'black',
        'fontWeight': 'bold'
    }
    ],
        page_size=10
    )

    # Tabella break even
    dfBE_display = dfBE.round(2).copy()
    dfBE_display["index"] = dfBE_display.index
    dfBE_display = dfBE_display[["index"] + [col for col in dfBE_display.columns if col != "index"]]
    
    # Crea tabella
    tableBE = dash.dash_table.DataTable(
        data=[
       {str(k): v for k, v in row.items()}
       for row in dfBE_display.to_dict("records")
   ],
   columns=[
       {"name": str(col), "id": str(col)}
       for col in df_display.columns
   ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        style_data_conditional=[
    {
        'if': {'column_id': 'index'},
        'backgroundColor': '#d6e9f9',  # azzurrino chiaro
        'color': 'black',
        'fontWeight': 'bold'
    }
    ],
        page_size=10
    )
    
    # Tabella expiries
    tableExpiries = dash.dash_table.DataTable(
        data=[
        {str(k): v for k, v in row.items()}
        for row in ExpireTable.to_dict("records")
        ],
        columns=[
        {"name": str(col), "id": str(col)}
        for col in ExpireTable.columns
        ],
   style_table={
        'overflowX': 'auto',
        'width': '15%',         # Limita la larghezza della tabella
        'margin': '0',          # Nessun margine
        'marginLeft': '0px'     # Allinea a sinistra
    },
    style_cell={
        'textAlign': 'center'     # Allinea il contenuto delle celle a sinistra
    },
    
        page_size=10
    )
    
    ##
    putcall = ["P", "C"]

    initial_rows = [
    {"Contratto": None, "Strike": None, "P/C": None, "Volume": None}
    for _ in range(4)
    ]

    tabella_input = html.Div([
        html.H4("Portfolio"),
        dash_table.DataTable(
            id='input-opzioni',
            columns=[
                {"name": "Contratto", "id": "Contratto", "presentation": "dropdown"},
                {"name": "Strike", "id": "Strike", "type": "numeric"},
                {"name": "P/C", "id": "P/C", "presentation": "dropdown"},
                {"name": "Volume", "id": "Volume", "type": "numeric"},
            ],
            data=initial_rows,
            editable=True,
            dropdown={
                "Contratto": {
                    "options": [{"label": str(i), "value": str(i)} for i in ExpireTable['Contracts'].dropna().unique()]
                },
                "P/C": {
                    "options": [{"label": i, "value": i} for i in putcall]
                },
            },
            style_table={'minWidth': '100%', 'overflowX': 'auto', 'zIndex': 1000},
            style_cell={'textAlign': 'center'},
            row_deletable=True
        )
    ])
    
    layout_tabella = html.Div([
        html.H4("Premium Surface"),
        table,
        html.H4("Break Even Surface"),
        tableBE,
        html.H4("Expire Table"),
        tableExpiries,
    ])



    # Dropdown parametri: es. colonna 'strike'
    # opzioni = [{"label": str(i), "value": i} for i in df['strike'].unique()]

    return layout_tabella, tabella_input, ExpireTable.to_dict('records')

@app.callback(
    Output('output-grafici', 'children'),
    Input('btn-grafici', 'n_clicks'),
    State('input-opzioni', 'data'),
    State('end-date', 'date'),
    State('data-osservazione', 'date'),
    State('store-expiries', 'data'),
    prevent_initial_call=True
)
def run_backtest(n_clicks, portfolio_rows, end_date, start_date, expiries):
    if not n_clicks or not portfolio_rows or not end_date:
        return []

    # Prepara dict delle scadenze
    expiry_dict = {e['Contracts']: e['Expiries'] for e in expiries}

    option_specs = []
    for row in portfolio_rows:
        if not row['Contratto'] or row['Strike'] is None or not row['P/C'] or row['Volume'] is None:
            continue
        
        name = f"{row['Contratto']} {row['P/C']}{row['Strike']}"
        option_type = 'call' if row['P/C'] == 'C' else 'put'
        position = row['Volume']
        expiry = expiry_dict.get(row['Contratto'], None)
        if not expiry:
            continue
        underlying = row['Contratto'].split()[1]
        
        option_specs.append({
            'name': name,
            'K': float(row['Strike']),
            'type': option_type,
            'position': int(position),
            'expiry': expiry,
            'underlying': underlying
        })

    # === Esegui lo script di backtest in una funzione separata ===
    figures = run_analysis(start_date, end_date, option_specs)

    # === Converte i grafici matplotlib in immagini base64 ===
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    immagini = []
    for fig in figures:
        encoded = fig_to_base64(fig)
        immagini.append(html.Img(src=f'data:image/png;base64,{encoded}', style={'width': '100%'}))

    return immagini

if __name__ == '__main__':
    app.run(debug=True)