import streamlit as st
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
from bokeh.plotting import figure
import os
import datetime 
from datetime import date
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, MaxPooling1D, MaxPooling2D 
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from functions import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

current_directory = os.path.dirname(os.path.realpath(__file__))
directory_immagine = os.path.join(current_directory, 'img/pignataro_patrimonio.png') 
im = Image.open(directory_immagine)
st.set_page_config(
    page_title="Air quality prediction",
    page_icon=im,
)

page = st.selectbox("Choose your page", ['PM10', 'PM2.5', 'PM1']) 

def apri_csv_PM10():
    
    '''Puoi caricare un csv con un drag and drop o selezionandolo manualmente'''
    
    data_file = st.file_uploader("Upload file to predict PM10", type=["xlsx"]) #crea un uploader per inserire il file .csv
    if st.button("Compute prediction"): #se viene premuto il tasto esegui... 
        if data_file is not None: #... e il df non è un None Type...
            df = pd.read_excel(data_file) #... legge il file...
            st.dataframe(df) #...viene printato il dataframe
            return df #ritorna il dataframe

def apri_csv_PM1():
    
    '''Puoi caricare un csv con un drag and drop o selezionandolo manualmente'''
    
    data_file = st.file_uploader("Carica il file per avere la predizione del PM1", type=["xlsx"]) #crea un uploader per inserire il file .csv
    if st.button("Compute prediction"): #se viene premuto il tasto esegui... 
        if data_file is not None: #... e il df non è un None Type...
            df = pd.read_excel(data_file) #... legge il file...
            st.dataframe(df) #...viene printato il dataframe
            return df #ritorna il dataframe

def apri_csv_PM25():
    
    '''Puoi caricare un csv con un drag and drop o selezionandolo manualmente'''
    
    data_file = st.file_uploader("Carica il file per avere la predizione del PM2.5", type=["xlsx"]) #crea un uploader per inserire il file .csv
    if st.button("Compute prediction"): #se viene premuto il tasto esegui... 
        if data_file is not None: #... e il df non è un None Type...
            df = pd.read_excel(data_file) #... legge il file...
            st.dataframe(df) #...viene printato il dataframe
            return df #ritorna il dataframe

def carica_modello(percorso_modello): 
    
    '''Apre un modello pre-creato e lo rende utilizzabile'''
    
    return tf.keras.models.load_model(percorso_modello, compile = False)  
    
def carica_immagine(percorso_immagine):
    
    '''Apre un'immagine in locale per rendere la web page esteticamente più gradevole'''
    
    img = Image.open(percorso_immagine)
    st.image(img, width = 200) 

def finestra_scorrevole(df, model, pm_list=[]):
    
    '''Funzione chiave. Rende la serie scorrevole: in input al modello andranno infatti i valori di Temperatura Media, Umidità e PM10 dei 7 giorni precedenti più i valori di Temperatura ed Umidità del giorno in cui si vuole prevedere il valore di inquinamento IN QUESTO PRECISO ORDINE. Dopodiché, per fare una previsione a 7 giorni la finestra diventa scorrevole perché vengono eliminati dal dataframe i primi tre valori e verranno aggiunti il valore previsto per l'inquinamento ed i valori di Temperatura Media ed Umidità del giorno successivo, che a loro volta sono stati presi dal csv dato in input al sistema'''
    
    numeri = [i for i in range (0,47)] #range preso dal modello in input volta per volta (c'era 23, poi 47)
    try:
        data = df.iloc[:, numeri] #iloc del dataframe per quella che è l'ampiezza della finestra
        data = data.values.reshape((data.shape[0], data.shape[1], 1))
    except:
        st.error("File must be in .xlsx format and must respect days and features order")
        st.stop()

    PM_previsto = np.round(model.predict(data)[0]) #primo predict: è quello fatto senza far mai 'scorrere' la finestra temporale

    pm_list.append(PM_previsto) #primo append alla lista vuota
    for i in range(6): #per sei volte, tanti quanti sono i giorni per 'riempire' la settimana
        copia = df.copy() #copia del dataframe in input. Serve per prendere i 'pezzi' per riaggiornare il dataframe ad ogni iterazione
        data = data[:, 6:] # ogni volta levo le prime 6 variabili
        data = pd.DataFrame(data.reshape(1, 41))
        df_1=pd.DataFrame(PM_previsto) #il PM previsto si trasforma in un oggetto pandas perché deve essere inserito nel df per poter essere utilizzato come input per calcolare il giorno seguente
        data = pd.concat([data, df_1], axis=1) #momento in cui si concatena il PM previsto al dataframe per essere usato come input per calcolare il livello di inquinamento del giorno successivo
        z = 0+i*2 #variabile che serve per fare scorrere di due posizioni la serie a destra
        e = 1+i*2 #uguale a z
        p = 2+i*2
        j = 3+i*2
        q = 4+i*2
        df_2 = copia.iloc[:,[23+z, 23+e, 23+p, 23+j, 23+q]]
        data = pd.concat([data, df_2], axis=1) #si concatenano al df i valori presi prima della copia e si crea di nuovo un df per poter fare prediction
        data = data.values.reshape((data.shape[0], data.shape[1], 1))
        PM_previsto = np.round(model.predict(data)[0]) #si approssima il valore predetto (i valori di PM sono sempre visti come interi)
        pm_list.append(PM_previsto) #append della prediction alla variabile locale pm_list

def lista_a_stringa(lista, misura):
    
    '''Permette di trasformare la lista in una stringa che compare sulla pagina'''
    
    pm_string = str(lista).replace('array','').replace('[','').replace(']','').replace(', dtype=float32', '').replace('(','').replace(')','').replace('.','') #trasforma e pulisce la lista in input
    st.header('Predicted ' + misura + ' values for next 7 days are: ' + pm_string) #printa i valori previsti
    return pm_string

def grafico_PM10(stringa):
    a_str = stringa.split(',') #splitta la stringa separando ogni qual volta vede una virgola
    lista = [float(i) for i in a_str] #crea una lista con i valori delle previsioni
    p = figure( #crea una figura
        title='Weekly PM10 predictions',
        x_axis_label='days',
        y_axis_label='PM10')
    giorni = [i for i in range(1,8)] #lista che viene utilizzata nell'asse delle ascisse
    #giorni = [(datetime.date.today() + datetime.timedelta(days=x)).strftime('%d-%m-%Y') for x in range(8)][1:]
    p.line(giorni, lista, legend_label='PM10 prediction', line_width=2) #crea la figura  
    st.bokeh_chart(p, use_container_width=True) #mostra la figura

def grafico_PM1(stringa):
    a_str = stringa.split(',') #splitta la stringa separando ogni qual volta vede una virgola
    lista = [float(i) for i in a_str] #crea una lista con i valori delle previsioni
    p = figure( #crea una figura
        title='Weekly PM1 prediction',
        x_axis_label='days',
        y_axis_label='PM1')
    giorni = [i for i in range(1,8)] #lista che viene utilizzata nell'asse delle ascisse
    p.line(giorni, lista, legend_label='PM1 prediction', line_width=2) #crea la figura  
    st.bokeh_chart(p, use_container_width=True) #mostra la figura

def grafico_PM25(stringa):
    a_str = stringa.split(',') #splitta la stringa separando ogni qual volta vede una virgola
    lista = [float(i) for i in a_str] #crea una lista con i valori delle previsioni
    p = figure( #crea una figura
        title='Weekly PM2.5 predictions',
        x_axis_label='days',
        y_axis_label='PM2.5')
    giorni = [i for i in range(1,8)] #lista che viene utilizzata nell'asse delle ascisse
    p.line(giorni, lista, legend_label='PM2.5 prediction', line_width=2) #crea la figura  
    st.bokeh_chart(p, use_container_width=True) #mostra la figura

def aggiorna_PM10(db, df):
    
    '''I valori (t-1) vanno aggiunti al db per poter fare il ricalcolo del modello.
    Sistema anche alcune specifiche del df'''

    current_directory = os.path.dirname(os.path.realpath(__file__))
    try:
        subset_da_aggiungere = df.iloc[:, 36:42]# i valori (t-1) hanno posizione fissa che va da 36 a 42
        oggi = date.today().strftime("%Y-%m-%d")
        subset_da_aggiungere['Wind Direction'] = subset_da_aggiungere['Wind_Angle(t)'].replace(360, 'N').replace(22.5, 'NNE').replace(45, 'NE').replace(67.5, 'ENE').replace(
            90, 'E').replace(112.5, 'ESE').replace(135, 'SE').replace(157.5, 'SSE').replace(180, 'S').replace(202.5, 'SSW').replace(225, 'SW').replace(
                247.5, 'WSW').replace(270, 'W').replace(292.5, 'WNW').replace(315, 'NW').replace(337.5, 'NNW')
        subset_da_aggiungere = subset_da_aggiungere.rename(columns = {'Humidity(t)': 'H', 'Wind_Speed(t)': 'Vento', 'Pressure(t)': 'Pressione', 'PM10(t)': 'PM10', 'Temperature(t)': 'T Media'})
        db = pd.concat((db, subset_da_aggiungere), axis = 0)
        db = db[db.datetime != oggi] # elimina la precedente riga con indice pari alla data odierna -> faccio così per avere soltanto l'ultima osservazione per quella data
        db.iloc[-1, 1] = oggi
        db = db.reset_index()
        db['datetime'] = pd.to_datetime(db['datetime']).dt.date #normalize()
        db.iloc[-1, 6] = str(db.iloc[-1, 6]) + ' km/h'  
        db = db.drop(columns = ['index', 'Wind_Angle(t)', 'Unnamed: 0'])
        db.to_excel(os.path.join(current_directory, 'storical/PM10.xlsx')) # aggiorno il db
    except:
        pass
    

def aggiorna_PM1(db, df):
    
    '''I valori (t-1) vanno aggiunti al db per poter fare il ricalcolo del modello.
    Sistema anche alcune specifiche del df'''

    current_directory = os.path.dirname(os.path.realpath(__file__))
    try:
        subset_da_aggiungere = df.iloc[:, 36:42]# i valori (t-1) hanno posizione fissa che va da 36 a 42
        oggi = date.today().strftime("%Y-%m-%d")
        subset_da_aggiungere['Wind Direction'] = subset_da_aggiungere['Wind_Angle(t)'].replace(360, 'N').replace(22.5, 'NNE').replace(45, 'NE').replace(67.5, 'ENE').replace(
            90, 'E').replace(112.5, 'ESE').replace(135, 'SE').replace(157.5, 'SSE').replace(180, 'S').replace(202.5, 'SSW').replace(225, 'SW').replace(
                247.5, 'WSW').replace(270, 'W').replace(292.5, 'WNW').replace(315, 'NW').replace(337.5, 'NNW')
        subset_da_aggiungere = subset_da_aggiungere.rename(columns = {'Humidity(t)': 'H', 'Wind_Speed(t)': 'Vento', 'Pressure(t)': 'Pressione', 'PM1(t)': 'PM1', 'Temperature(t)': 'T Media'})
        db = pd.concat((db, subset_da_aggiungere), axis = 0)
        db = db[db.datetime != oggi] # elimina la precedente riga con indice pari alla data odierna -> faccio così per avere soltanto l'ultima osservazione per quella data
        db.iloc[-1, 1] = oggi
        db = db.reset_index()
        db['datetime'] = pd.to_datetime(db['datetime']).dt.date #normalize()
        db.iloc[-1, 6] = str(db.iloc[-1, 6]) + ' km/h'  
        db = db.drop(columns = ['index', 'Wind_Angle(t)', 'Unnamed: 0'])
        db.to_excel(os.path.join(current_directory, 'storical/PM1.xlsx')) # aggiorno il db
    except:
        pass

def aggiorna_PM2_5(db, df):
    
    '''I valori (t-1) vanno aggiunti al db per poter fare il ricalcolo del modello.
    Sistema anche alcune specifiche del df'''

    current_directory = os.path.dirname(os.path.realpath(__file__))
    try:
        subset_da_aggiungere = df.iloc[:, 36:42]# i valori (t-1) hanno posizione fissa che va da 36 a 42
        oggi = date.today().strftime("%Y-%m-%d")
        subset_da_aggiungere['Wind Direction'] = subset_da_aggiungere['Wind_Angle(t)'].replace(360, 'N').replace(22.5, 'NNE').replace(45, 'NE').replace(67.5, 'ENE').replace(
            90, 'E').replace(112.5, 'ESE').replace(135, 'SE').replace(157.5, 'SSE').replace(180, 'S').replace(202.5, 'SSW').replace(225, 'SW').replace(
                247.5, 'WSW').replace(270, 'W').replace(292.5, 'WNW').replace(315, 'NW').replace(337.5, 'NNW')
        subset_da_aggiungere = subset_da_aggiungere.rename(columns = {'Humidity(t)': 'H', 'Wind_Speed(t)': 'Vento', 'Pressure(t)': 'Pressione', 'PM2_5(t)': 'PM2_5', 'Temperature(t)': 'T Media'})
        db = pd.concat((db, subset_da_aggiungere), axis = 0)
        db = db[db.datetime != oggi] # elimina la precedente riga con indice pari alla data odierna -> faccio così per avere soltanto l'ultima osservazione per quella data
        db.iloc[-1, 1] = oggi
        db = db.reset_index()
        db['datetime'] = pd.to_datetime(db['datetime']).dt.date #normalize()
        db.iloc[-1, 6] = str(db.iloc[-1, 6]) + ' km/h'  
        db = db.drop(columns = ['index', 'Wind_Angle(t)', 'Unnamed: 0'])
        db.to_excel(os.path.join(current_directory, 'storical/PM2_5.xlsx')) # aggiorno il db
    except:
        pass

def lstm_PM10(X_train_series, Y_train, X_valid_series, Y_valid, db):
    '''Ogni 7 giorni ricalcola il modello'''
    current_directory = os.path.dirname(os.path.realpath(__file__))
    epochs = 350
    batch = 64
    lr = 0.0001
    adam = optimizers.Adam(lr)
    model_lstm = Sequential()
    model_lstm.add(LSTM(250, return_sequences=True)) 
    model_lstm.add(Dense(150))
    model_lstm.add(LSTM(50, activation='relu'))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    with st.spinner('**Updating model. Please wait...**'):
        model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
        model_lstm.save('modello_lstm_PM10.h5')
    st.success('**Model successfully updated**')
    return

def lstm_PM1(X_train_series, Y_train, X_valid_series, Y_valid, db):
    '''Ogni 7 giorni ricalcola il modello'''
    current_directory = os.path.dirname(os.path.realpath(__file__))
    epochs = 550
    batch = 64
    lr = 0.0001
    adam = optimizers.Adam(lr)
    model_lstm = Sequential()
    model_lstm.add(LSTM(250, return_sequences=True)) 
    model_lstm.add(Dense(150))
    model_lstm.add(LSTM(50, activation='relu'))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    with st.spinner('**Updating model. Please wait...**'):
        model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
        model_lstm.save('modello_lstm_PM1.h5')
    st.success('**Model successfully updated**')
    return

def lstm_PM2_5(X_train_series, Y_train, X_valid_series, Y_valid, db):
    '''Ogni 7 giorni ricalcola il modello'''
    current_directory = os.path.dirname(os.path.realpath(__file__))
    epochs = 550
    batch = 64
    lr = 0.0001
    adam = optimizers.Adam(lr)
    model_lstm = Sequential()
    model_lstm.add(LSTM(250, return_sequences=True)) 
    model_lstm.add(Dense(150))
    model_lstm.add(LSTM(50, activation='relu'))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    with st.spinner('**Updating model. Please wait...**'):
        model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=2)
        model_lstm.save('modello_lstm_PM2_5.h5')
    st.success('**Model successfully updated**')
    return

def weekly_quality_PM10(stringa_valori):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    percorso_immagine_verde = os.path.join(current_directory, 'img/verde.png') 
    percorso_immagine_giallo = os.path.join(current_directory, 'img/giallo.png')
    percorso_immagine_rosso = os.path.join(current_directory, 'img/rosso.png')
    img_verde = Image.open(percorso_immagine_verde)
    img_verde.convert('RGB').save('img/verde.png')
    img_giallo = Image.open(percorso_immagine_giallo)
    img_giallo.convert('RGB').save('img/giallo.png')
    img_rosso = Image.open(percorso_immagine_rosso)
    img_rosso.convert('RGB').save('img/rosso.png')
    tupla_valori = tuple(map(int, stringa_valori.split(', ')))
    col1, mid, col2 = st.beta_columns([5,1,5])
    with col1:
        st.write(commento_previsione_PM10(tupla_valori))
    with col2:
        if commento_previsione_PM10(tupla_valori) == "Weekly air quality is within the limits of the law":
            st.image(img_verde, width = 60)
        if commento_previsione_PM10(tupla_valori) == "Air quality is discreet but without big peaks":
            st.image(img_giallo, width = 60)
        if commento_previsione_PM10(tupla_valori) == "Expected weekly air quality is bad":
            st.image(img_rosso, width = 60)

def weekly_quality_PM2_5(stringa_valori):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    percorso_immagine_verde = os.path.join(current_directory, 'img/verde.png') 
    percorso_immagine_giallo = os.path.join(current_directory, 'img/giallo.png')
    percorso_immagine_rosso = os.path.join(current_directory, 'img/rosso.png')  
    img_verde = Image.open(percorso_immagine_verde)
    img_verde.convert('RGB').save('img/verde.png')
    img_giallo = Image.open(percorso_immagine_giallo)
    img_giallo.convert('RGB').save('img/giallo.png')
    img_rosso = Image.open(percorso_immagine_rosso)
    img_rosso.convert('RGB').save('img/rosso.png')
    tupla_valori = tuple(map(int, stringa_valori.split(', ')))
    col1, mid, col2 = st.beta_columns([5,1,5])
    with col1:
        st.write(commento_previsione_PM25(tupla_valori))
    with col2:
        if commento_previsione_PM25(tupla_valori) == "PM2.5 thresholds will be respected every day":
            st.image(img_verde, width = 60)
        if commento_previsione_PM25(tupla_valori) == "High PM2.5 values":
            st.image(img_rosso, width = 60)
        if commento_previsione_PM25(tupla_valori) == "Air quality is discreet but without big peaks":
             st.image(img_giallo, width = 60)

def PM10():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    directory_immagine = os.path.join(current_directory, 'img/pignataro_patrimonio.png')   
    directory_modello = os.path.join(current_directory, 'modello_lstm_PM10.h5') 
    db = pd.read_excel(os.path.join(current_directory, 'storical/PM10.xlsx'))
    pm_inquinamento = []
    csv = apri_csv_PM10() #apre un csv e lo rende utilizzabile per il nostro lavoro
    modello = carica_modello(directory_modello)
    carica_immagine(directory_immagine)
    aggiorna_PM10(db, csv)
    finestra_scorrevole(csv, modello, pm_inquinamento)
    stringa_inquinamento = lista_a_stringa(pm_inquinamento, 'PM10')
    weekly_quality_PM10(stringa_inquinamento)
    grafico_PM10(stringa_inquinamento)
    db = pd.read_excel(os.path.join(current_directory, 'storical/PM10.xlsx')) # riapro il file che adesso è aggiornato
    st.header(superamenti_anno_PM10(db)) # 
    if db.shape[0] >= 365 + 7: # perché voglio l'aggiornamento dopo la prima settimana
        if (db.shape[0] - 365) % 7 == 0:
            X_train_series, Y_train, X_valid_series, Y_valid = train_val_df(db, 'PM10')
            lstm_PM10(X_train_series, Y_train, X_valid_series, Y_valid, db)
    
def PM1():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    directory_modello = os.path.join(current_directory, 'modello_lstm_PM1.h5') 
    directory_immagine = os.path.join(current_directory, 'img/pignataro_patrimonio.png') 
    db = pd.read_excel(os.path.join(current_directory, 'storical/PM1.xlsx'))
    pm_inquinamento = []
    csv = apri_csv_PM1() #apre un csv e lo rende utilizzabile per il nostro lavoro
    modello = carica_modello(directory_modello)
    carica_immagine(directory_immagine)
    aggiorna_PM1(db, csv)
    finestra_scorrevole(csv, modello, pm_inquinamento)
    stringa_inquinamento = lista_a_stringa(pm_inquinamento, 'PM1')
    grafico_PM1(stringa_inquinamento)
    if db.shape[0] >= 365 + 7: # perché voglio l'aggiornamento dopo la prima settimana
        if (db.shape[0] - 365) % 7 == 0:
            X_train_series, Y_train, X_valid_series, Y_valid = train_val_df(db, 'PM1')
            lstm_PM1(X_train_series, Y_train, X_valid_series, Y_valid, db)

def PM25():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    directory_modello = os.path.join(current_directory, 'modello_lstm_PM2_5.h5') 
    directory_immagine = os.path.join(current_directory, 'img/pignataro_patrimonio.png') 
    db = pd.read_excel(os.path.join(current_directory, 'storical/PM2_5.xlsx'))
    pm_inquinamento = []
    csv =  apri_csv_PM25() #apre un csv e lo rende utilizzabile per il nostro lavoro
    modello = carica_modello(directory_modello)
    carica_immagine(directory_immagine)
    aggiorna_PM2_5(db, csv)
    finestra_scorrevole(csv, modello, pm_inquinamento)
    stringa_inquinamento = lista_a_stringa(pm_inquinamento, 'PM2.5')
    weekly_quality_PM2_5(stringa_inquinamento)
    grafico_PM25(stringa_inquinamento)
    if db.shape[0] >= 365 + 7: # perché voglio l'aggiornamento dopo la prima settimana
        if (db.shape[0] - 365) % 7 == 0:
            X_train_series, Y_train, X_valid_series, Y_valid = train_val_df(db, 'PM2_5')
            lstm_PM2_5(X_train_series, Y_train, X_valid_series, Y_valid, db)

if __name__ == '__main__':
    if page == 'PM10':
        PM10()
    elif page == 'PM2.5':
        PM25()
    else:
        PM1()
    