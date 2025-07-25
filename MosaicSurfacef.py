# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:08:08 2025

@author: GCicerani
"""
def Surface(Date, BSCheck):
    import os
    os.environ['MOSAIC_ENV'] = 'PROD'
    from mosaic_api_client.options_api import OptionsApi as OA
    import pandas as pd
    import numpy as np
    from scipy.stats import norm
    holidays = []
    
    
    ##########################################################################
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
    
    def BachelierEurope(F, K, vol, Ttm, r):
        d1 = (F - K) / (vol * np.sqrt(Ttm))
        C = np.exp(-r * Ttm) * ((F - K) * norm.cdf(d1) + (vol * (np.sqrt(Ttm) / np.sqrt(2 * np.pi)) * np.exp(-d1 ** 2 / 2)))
        P = np.exp(-r * Ttm) * ((K - F) * norm.cdf(-d1) + (vol * (np.sqrt(Ttm) / np.sqrt(2 * np.pi)) * np.exp(-d1 ** 2 / 2)))
        return C, P
    
    def working_days_to_expiry(current_date, target_expiry):
        bdays = pd.bdate_range(current_date, target_expiry, freq='C', holidays=holidays)
        return max(len(bdays) - 1, 0)
    
    
    def BlackScholesPrice(F, K, vol, Ttm, r, option_type):
        # F: forward price (underlying)
        # K: strike
        # vol: volatility (annualized std)
        # Ttm: time to maturity (in years)
        # r: interest rate (can essere 0 se usi forward)
        # option_type: 'call' or 'put'
        d1 = (np.log(F / K) + 0.5 * vol**2 * Ttm) / (vol * np.sqrt(Ttm))
        d2 = d1 - vol * np.sqrt(Ttm)
        discount = np.exp(-r * Ttm)
        if option_type == 'call':
            price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        return price
    
    
    def ImpliedVol(F, K, BSIniziale, Ttm, r, Ptarget, Type):
        NewBS = BSIniziale
        # Calcolo iniziale differenza tra prezzo target e prezzo BS
        DiffPrices = Ptarget - BlackScholesPrice(F, K, BSIniziale, Ttm, r, Type)
        count = 0
        while abs(DiffPrices) > 0.02:  # tolleranza di 2 centesimi
            P0 = BlackScholesPrice(F, K, NewBS, Ttm, r, Type)
            Pvup = BlackScholesPrice(F, K, NewBS + 0.001, Ttm, r, Type)
            Vega = Pvup - P0
            if Vega == 0:
                break  # evita divisione per zero
            DiffVol = DiffPrices / Vega * 0.001  # 0.01 = passo usato
            NewBS += DiffVol
            DiffPrices = Ptarget - BlackScholesPrice(F, K, NewBS, Ttm, r, Type)
            count += 1
            if count > 10:
                break
        return NewBS
    
    ##########################################################################
    #Date = '2025-06-19'
    
    # Importo superficie di volatilita' a quella data
    MosaiCSurface = OA.get_option_surface_settlement(symbol='B', exchange='ICE', 
                                    as_of_date= Date, allow_indicative=True, include_oi=True)
    
    # Contracts
    Contracts = MosaiCSurface['future_key'].unique()
    
    # Expiries
    Expiries = MosaiCSurface['expiration_date'].unique()
    ExpireTable = pd.DataFrame({
        'Contracts': Contracts,
        'Expiries': Expiries})
    
    # Underlying
    S = MosaiCSurface['future_value'].unique()
    
    # Creo vettore di strike
    S0 = S[0]
    Srounded = round(S0/5) * 5
    Strikes = np.arange(Srounded-25,Srounded+30,5)
    
    # Riempio la superficie dei premi
    PremiumSurface = np.zeros([12, len(Strikes)])
    
    for i in range(12):
        contract = Contracts[i]
        underlyng = S[i]    
        for j in range(len(Strikes)):
            strike = Strikes[j]
            typ = 'P' if strike < underlyng else 'C'
            key = contract + ' ' + typ + str(strike)
            Idx = MosaiCSurface['instrument_key'] == key
            PremiumSurface[i,j] = MosaiCSurface.loc[Idx, 'value'].item()
            
    
    # Creazione Dataframe
    columns = Strikes.tolist() + ['underlying']
    full_matrix = np.hstack([PremiumSurface, np.array(S[0:12]).reshape(-1,1)])
    # Creazione DataFrame
    df = pd.DataFrame(full_matrix, index=Contracts[0:12], columns=columns)
    
    # Superfice implied breakeven
    BESurface = np.zeros([12, len(Strikes)])
    # breakpoint()
    for i in range(12):
        contract = Contracts[i]
        underlyng = S[i]
        for j in range(len(Strikes)):
            strike = Strikes[j]
            typ = 'P' if strike < underlyng else 'C'
            premium = PremiumSurface[i,j]
            expiry = ExpireTable.loc[ExpireTable['Contracts'] == contract, 'Expiries'].item()
            days = working_days_to_expiry(Date, expiry)
            Ttm = days/255
            Type = 'call' if typ == 'C' else 'put'
            if BSCheck == True:
                be = ImpliedVol(underlyng, strike, 0.6, Ttm, 0, premium, Type)                
            else:
                be = BEImplicito(underlyng, strike, 1.50, Ttm, 0, premium, Type)
            BESurface[i,j] = be
    fullmatrix_be = np.hstack([BESurface, np.array(S[0:12]).reshape(-1,1)])
    dfBE = pd.DataFrame(fullmatrix_be, index=Contracts[0:12], columns=columns)

    
    return df, dfBE, ExpireTable

