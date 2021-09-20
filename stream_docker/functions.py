import pandas as pd
import numpy as np
import datetime
from bokeh.plotting import figure

def series_to_supervised(data, window=1, lag=1, dropnan=True):

    '''Aggiunge il desiderato numero di giorni precedenti e successivi al giorno t'''

    cols, names = list(), list()
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def train_val_df(dataset_finale, labels_col):

    '''Calcola X_train_series, Y_train, X_valid_series, Y_valid'''

    dataset_finale['Vento'] = dataset_finale['Vento'].map(lambda x: x.rstrip(' km/h')) # elimino il ' km/' 
    dataset_finale = dataset_finale.astype({"Vento": int}) # trasformo in int
    dataset_finale['Wind_Angle'] = dataset_finale['Wind Direction'].replace('N', 360).replace('NNE', 22.5).replace('NE', 45).replace('ENE', 67.5).replace(
        'E', 90).replace('ESE', 112.5).replace('SE', 135).replace('SSE', 157.5).replace('S', 180).replace('SSW', 202.5).replace('SW', 225).replace(
            'WSW', 247.5).replace('W', 270).replace('WNW', 292.5).replace('NW', 315).replace('NNW', 337.5)
    particolato = dataset_finale[labels_col]
    df = dataset_finale.drop(columns = ['Wind Direction', str(labels_col)])
    df = pd.concat((df, particolato), axis = 1)
    df = df.rename(columns = {'T Media': 'Temperature', 'H': 'Humidity', 'Vento': 'Wind_Speed', 'Pressione': 'Pressure'})
    window = 6
    lag = 1
    series = series_to_supervised(df.drop('datetime', axis=1), window=window, lag=lag)
    n = len(series)
    labels_col = labels_col # 'PM10(t)'
    labels = series[labels_col + '(t)']
    series = series.drop(labels_col + '(t)', axis=1)
    X_train = series[0:int(n*0.8)] # (981) per avere gli ultimi 7 giorni come valid # per avere la prediction sugli ultimi 6 giorni moltiplica per 0.984, 0.960 per gli ultimi 15 giorni
    X_valid = series[int(n*0.8):]
    Y_train = labels[0:int(n*0.8)]
    Y_valid = labels[int(n*0.8):]
    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    return X_train_series, Y_train.to_numpy(), X_valid_series, Y_valid.to_numpy()

def commento_previsione_PM10(tupla_di_valori):

    '''Commenta la previsione settimanale considerando quelli che sono i limiti nazionali alla concentrazione di PM10'''
    
    contatore = 0
    for valore in tupla_di_valori: # caso in cui tutti i valori sono sotto 40
        if valore < 40:
            contatore += 1
    if contatore == len(tupla_di_valori):
        frase_da_ritornare = "Weekly air quality is within the limits of the law"
    elif np.mean(tupla_di_valori) > 40 and np.mean(tupla_di_valori) < 70:
        frase_da_ritornare = "Air quality is discreet but without big peaks"
    elif np.mean(tupla_di_valori) > 70:
        frase_da_ritornare = "Expected weekly air quality is bad"
    else: 
        frase_da_ritornare = "Air quality is discreet but without big peaks"
    return frase_da_ritornare

def commento_previsione_PM25(tupla_di_valori):
    
    '''Commenta la previsione settimanale considerando quelli che sono i limiti nazionali alla concentrazione di PM10'''
    
    contatore = 0
    for valore in tupla_di_valori: # caso in cui tutti i valori sono sotto 40
        if valore < 26:
            contatore += 1
    if contatore == len(tupla_di_valori):
        frase_da_ritornare = "PM2.5 thresholds will be respected every day"
    elif np.mean(tupla_di_valori) > 10 and np.mean(tupla_di_valori) < 40:
        frase_da_ritornare = "Air quality is discreet but without big peaks"
    else: 
        frase_da_ritornare = "High PM2.5 values"
    return frase_da_ritornare

def superamenti_anno_PM10(db):

    '''Conta i giorni nell'anno solare in cui è stato superato il limite di 50 µg/m3. 
    Per legge in Italia non può essere superato più di 35 volte'''

    contatore = 0
    current_year = datetime.datetime.now().year
    db_anno_corrente = db[db['datetime'].dt.year == current_year]
    for value in db_anno_corrente.PM10:
        if value > 50:
            contatore += 1
    return "Limit of 50 µg/m3 this year has been exceeded " + str(contatore) + " times (max is 35)"