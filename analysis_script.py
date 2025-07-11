def run_analysis(start_date, end_date, option_specs):

    import os
    os.environ['MOSAIC_ENV'] = 'PROD'
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import timedelta
    import matplotlib.dates as mdates
    from matplotlib import style
    style.use('seaborn-v0_8-paper')
    from scipy.stats import norm
    from mosaic_api_client.options_api import OptionsApi as OA
    
    
    ############################## INPUT ##########################################
    
   
    holidays = pd.to_datetime(['16-April-2025'])
    Underlyings = [d['underlying'] for d in option_specs if 'underlying' in d]
    UniqueUnderlyings = list(set(Underlyings))
    BEIniziale = 1.30
    OptionNames = [n['name'] for n in option_specs]
    TotalNames = OptionNames + UniqueUnderlyings
    title = 'My ptf'
    
    ############################ Import data from mosaic #########################
    
    # Dates
    bdays = pd.bdate_range(start_date, end_date, freq='C', holidays=holidays)
    Dates = bdays
    
    
    # Main loop to get data
    Timeseries = {ts: np.zeros(len(bdays)) for ts in TotalNames}
    
    for i in range(len(bdays)):
        for c in UniqueUnderlyings:
                MosaicSLice = OA.get_option_slice_settlement(symbol='B', exchange='ICE', term= c,
                                    as_of_date= bdays[i].date())
                Timeseries[c][i] = MosaicSLice.loc[0,'future_value']
                # Get Option specs with c underlying
                opts = [o for o in option_specs if o['underlying'] == c]
                for opt in opts:
                    name = opt['name']
                    idx = MosaicSLice['instrument_key'] == name
                    settlement = MosaicSLice.loc[idx, 'value'].iloc[0]
                    Timeseries[name][i] = settlement
    
    Timeseries = {'Date:': bdays, **Timeseries}
    Data = pd.DataFrame(Timeseries)
    
    ############################# Functions ######################################
    
    def BachelierEurope(F, K, vol, Ttm, r):
        d1 = (F - K) / (vol * np.sqrt(Ttm))
        C = np.exp(-r * Ttm) * ((F - K) * norm.cdf(d1) + (vol * (np.sqrt(Ttm) / np.sqrt(2 * np.pi)) * np.exp(-d1 ** 2 / 2)))
        P = np.exp(-r * Ttm) * ((K - F) * norm.cdf(-d1) + (vol * (np.sqrt(Ttm) / np.sqrt(2 * np.pi)) * np.exp(-d1 ** 2 / 2)))
        return C, P
    
    def DeltaBachelierEurope(F, K, vol, Ttm, r, shock):
        # P0 = np.array(BachelierEurope(F, K, vol, Ttm, r))
        Pup = np.array(BachelierEurope(F+ shock, K, vol, Ttm, r))
        Pdown = np.array(BachelierEurope(F- shock, K, vol, Ttm, r))
        delta = (Pup-Pdown)/(2*shock)
        return delta
    
    def GammaBachelierEurope(S, K, vol, T, r, shock):
        P0 = BachelierEurope(S, K, vol, T, r)
        PriceUp = np.array(BachelierEurope(S+shock, K, vol, T, r))
        PriceDown = np.array(BachelierEurope(S-shock, K, vol, T, r))
        return (PriceUp + PriceDown - 2 * np.array(P0)) / (shock ** 2)
    
    def ThetaBachelierEurope(tipo, S, K, T, r, vol, DaysYear):
        P0 = BachelierEurope(S, K, vol, T, r)
        Tstar = T - 1 / DaysYear
        if Tstar > 0:
            Ptomorrow = BachelierEurope(S, K, vol, Tstar, r)
        else:
            Ptomorrow = BachelierEurope( S, K, vol, 1e-7, r)
        if tipo == 'call':
            return Ptomorrow[0] - P0[0]
        else:
            return Ptomorrow[1] - P0[1]
    
    def VegaBachelierEurope(S, K, vol, T, r, shock):
        P0 = np.array(BachelierEurope(S, K, vol, T, r))
        Pvega = np.array(BachelierEurope(S, K, vol+shock*np.sqrt(255), T, r))
        return Pvega - P0
    
    def VannaBachelierEurope(S, K, vol, T, r, shock, shockVol):
        DeltaUpVolUp = np.array(DeltaBachelierEurope(S, K, vol+shockVol, T, r, shock))
        Delta = np.array(DeltaBachelierEurope(S, K, vol, T, r, shock))
        return DeltaUpVolUp - Delta
    
    
    def working_days_to_expiry(current_date, target_expiry):
        bdays = pd.bdate_range(current_date, target_expiry, freq='C', holidays=holidays)
        return max(len(bdays) - 1, 0)
    
    # days_to_expiry = Dates.map(working_days_to_expiry)
    # Ttm = days_to_expiry / 255
    
    # === FUNCTIONS ===
    def DeltaBachelier(F, K, vol, Ttm, r):
        Pup = np.array(BachelierEurope(F + 0.01, K, vol, Ttm, r))
        Pdown = np.array(BachelierEurope(F - 0.01, K, vol, Ttm, r))
        delta = (Pup - Pdown) / 0.02
        return delta
    
    def BEImplicito(F, K, BEIniziale, Ttm, r, Ptarget, Type):
        NewBE = BEIniziale
        DiffPrices = Ptarget - (BachelierEurope(F, K, BEIniziale * np.sqrt(255), Ttm, r)[0 if Type == 'call' else 1])
        count = 0
        while abs(DiffPrices) > 0.02:
            P0 = BachelierEurope(F, K, NewBE * np.sqrt(255), Ttm, r)
            Pvup = BachelierEurope(F, K, (NewBE + 0.01) * np.sqrt(255), Ttm, r)
            Vega = (Pvup[0 if Type == 'call' else 1]) - (P0[0 if Type == 'call' else 1])
            DiffBE = DiffPrices / Vega / 100
            NewBE += DiffBE
            DiffPrices = Ptarget - (BachelierEurope(F, K, NewBE * np.sqrt(255), Ttm, r)[0 if Type == 'call' else 1])
            count += 1
            if count >10:
                break
        return NewBE
    
    ############################# Main loop ######################################
    
    Greeks = {name: {g: np.zeros(len(Dates)) for g in ['BE', 'Delta', 'Gamma', 'Vega', 'Theta', 'Vanna']} for name in [o['name'] for o in option_specs]}
    PtfGreeks = {g: np.zeros(len(Dates)) for g in ['Delta', 'Gamma', 'Vega', 'Theta', 'Vanna']}
    for i, date in enumerate(Dates):
        for opt in option_specs:
            name, K, typ, pos, under = opt['name'], opt['K'], opt['type'], opt['position'], opt['underlying']
            S = Data[under]
            price = Data[name].iloc[i]
            expiry_date = pd.to_datetime(opt['expiry'])
            days_to_exp = working_days_to_expiry(date, expiry_date)
            Ttm = days_to_exp / 255
            be = BEImplicito(S.iloc[i], K, BEIniziale, Ttm, 0, price, typ)
            vol = be * np.sqrt(255)
            delta = DeltaBachelier(S.iloc[i], K, vol, Ttm, 0)[0 if typ == 'call' else 1]
            Greeks[name]['BE'][i] = be
            Greeks[name]['Delta'][i] = delta
            Greeks[name]['Gamma'][i] = GammaBachelierEurope(S.iloc[i], K, vol, Ttm, 0, 0.01)[0]
            Greeks[name]['Vega'][i] = VegaBachelierEurope(S.iloc[i], K, vol, Ttm, 0, 0.01)[0]
            Greeks[name]['Theta'][i] = ThetaBachelierEurope(typ, S.iloc[i], K, Ttm, 0, vol, 255)
            Greeks[name]['Vanna'][i] = VannaBachelierEurope(S.iloc[i], K, vol, Ttm, 0, 0.01, 0.01 * np.sqrt(255))[0]
    
            PtfGreeks['Delta'][i] += delta * pos
            PtfGreeks['Gamma'][i] += Greeks[name]['Gamma'][i] * pos
            PtfGreeks['Vega'][i] += Greeks[name]['Vega'][i] * pos * 1000
            PtfGreeks['Theta'][i] += Greeks[name]['Theta'][i] * pos * 1000
            PtfGreeks['Vanna'][i] += Greeks[name]['Vanna'][i] * pos * 1000
    
    ###############################  DELTA P&L ###################################
    
    DeltaTot = np.diff(PtfGreeks['Delta'])
    DeltaTot = np.concatenate([[0], DeltaTot])
    DeltaPL = np.zeros(len(Dates)-1)
    for opt in option_specs:
        S = Data[opt['underlying']]
        name, pos = opt['name'], opt['position']
        DeltaPL += -Greeks[name]['Delta'][:-1] * np.diff(S) * 1000 * pos
    for opt in option_specs:
        name = opt['name']
        DeltaPL += np.diff(Data[name]) * opt['position'] * 1000
    PL = np.cumsum(DeltaPL)
    
    ################################ GREEK EXPLANATION ###########################
    # DeltaPLfromGamma = 0.5 * np.diff(S)**2 * PtfGreeks['Gamma'][:-1] * 1000
    DeltaPLfromGamma = sum([
        np.diff(Data[opt['underlying']])** 2* 0.5 * Greeks[opt['name']]['Gamma'][:-1] * 1000  * opt['position']
        for opt in option_specs
    ])
    DeltaPLfromVega = sum([
        np.diff(Greeks[opt['name']]['BE']) * Greeks[opt['name']]['Vega'][:-1] * 1000 * 100 * opt['position']
        for opt in option_specs
    ])
    DeltaPLfromTheta = PtfGreeks['Theta'][:-1]
    DeltaPLfromVanna = sum([
        np.diff(Data[opt['underlying']]) * np.diff(Greeks[opt['name']]['BE']) * Greeks[opt['name']]['Vanna'][:-1] * 1000 * 100 * opt['position']
        for opt in option_specs
    ])
    DeltaPLfromGreeks = DeltaPLfromGamma + DeltaPLfromVega + DeltaPLfromTheta + DeltaPLfromVanna
    DeltaPLUnexplained = DeltaPL - DeltaPLfromGreeks
    
    ####################################### PLOT ##################################
    
    # === Analisi grafica generale ===
    
    fig1, ax1 = plt.subplots()
    for i in range(len(UniqueUnderlyings)):
        S = Data[UniqueUnderlyings[i]]
        ax1.plot(Dates, S, label= UniqueUnderlyings[i])
    ax1.set_title(f'{title} date: {start_date}')
    ax1.set_ylabel('Underlying (S)', color='#131f58')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = ax1.twinx()
    ax2.plot(Dates, np.concatenate([[0], PL]), color='#DE2A36')
    ax2.set_ylabel('P&L', color='#DE2A36')
    
    ax3 = ax1.twinx()
    ax3.spines['left'].set_position(('outward', 60))
    ax3.spines['left'].set_visible(True)
    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')
    
    ax3.plot(Dates, Greeks[option_specs[0]['name']]['BE'] ,color='#2c3e50', linestyle='--', label='Break Even')
    ax3.set_ylabel('Break Even')
    
    fig1.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
    
    # === DeltaS e BE ===
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    
    # Delta S
    bar_width = 0.2
    for i in range(len(UniqueUnderlyings)):
        S = Data[UniqueUnderlyings[i]]
        shift = timedelta(days=(i - len(UniqueUnderlyings)/2) * 0.2)
        ax1.bar(Dates[1:] + shift, np.diff(S), label = UniqueUnderlyings[i], width=bar_width)
        ax1.set_title('Delta S')
        ax1.grid(True)
        ax1.set_axisbelow(True)
        ax1.legend()
    
    # Delta Break Even per ogni opzione
    colors = plt.cm.tab10.colors  # palette fino a 10 colori
    delta = timedelta(days=0.2)
    
    
    for i in range(len(option_specs)):
        name = option_specs[i]['name']
        BE = Greeks[name]['BE']  # BreakEvenDict è un dizionario con chiavi=nome opzione, valori=serie break-even
        shift = timedelta(days=(i - len(option_specs)/2) * 0.2)
        ax2.bar(Dates[1:] + shift, np.diff(BE), label=f'Delta BE {name}', width=bar_width, color=colors[i % len(colors)])
    
    ax2.set_title('Delta Implied Break Even')
    ax2.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # === Greeks ptf ===
    
    figGreeks, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    greeks = {
        "Gamma": PtfGreeks['Gamma'],
        "Vega": PtfGreeks['Vega'],
        "Theta": PtfGreeks['Theta'],
        "Vanna": PtfGreeks['Vanna']
    }
    colors = ['#131f58', '#8c1c13', '#175e17', '#783c96']
    
    for i, (greek, series) in enumerate(greeks.items()):
        axes[i].bar(Dates[:-1], series[:-1], color=colors[i])
        axes[i].set_title(greek)
        axes[i].grid(True)
        axes[i].set_axisbelow(True)
    
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # === P&L explained ptf ===
    
    
    DeltaPLUnexplained = DeltaPL - DeltaPLfromGreeks
    
    x = Dates[1:]
    x1 = x - timedelta(days=0.3)
    x2 = x
    x3 = x + timedelta(days=0.3)
    
    figPnL, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.3
    
    # Serie A
    ax.bar(x1, DeltaPL, width=bar_width, label='Delta P&L', color='steelblue')
    
    # Serie B + C
    ax.bar(x2, DeltaPLfromGreeks, width=bar_width, label='DeltaP&L from greeks', color='darkorange')
    for i in range(len(x)):
        if DeltaPLUnexplained[i]*DeltaPLfromGreeks[i] >= 0:
            ax.bar(x2[i], DeltaPLUnexplained[i], bottom=DeltaPLfromGreeks[i], width=bar_width, color='lightsalmon', label='Unexplained' if i == 0 else "")
        else:
            ax.bar(x2[i], DeltaPLUnexplained[i], bottom=0, width=bar_width, color='lightsalmon', label='Unexplained' if i == 0 else "")
    
    # Serie D-G
    colors = ['#131f58', '#8c1c13', '#175e17', '#783c96']
    labels = ['Gamma', 'Vega', 'Theta', 'Vanna']
    components = [DeltaPLfromGamma, DeltaPLfromVega, DeltaPLfromTheta, DeltaPLfromVanna]
    
    for i in range(len(x)):
        pos_bottom = 0
        neg_bottom = 0
        for j, comp in enumerate(components):
            val = comp[i]
            if val >= 0:
                ax.bar(x3[i], val, bottom=pos_bottom, width=bar_width, color=colors[j], label=f'P&L from {labels[j]}' if i == 0 else "")
                pos_bottom += val
            else:
                ax.bar(x3[i], val, bottom=neg_bottom, width=bar_width, color=colors[j], label=f'P&L from {labels[j]}' if i == 0 else "")
                neg_bottom += val
    
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in x], rotation=45)
    ax.set_title('Delta P&L Explained')
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
    
    return [fig1, fig, figGreeks, figPnL]