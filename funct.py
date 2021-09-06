import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sesd

# def iterate_sep():
#     cwd = os.getcwd()
#     sep = '/Podatki SEP2'
# #     dictt = {}
#     dict_mismart = read_mismart(cwd+'/Mismart')
#     mismart = pd.DataFrame(dict_mismart)
#     for folder in os.listdir():
#         dictt, filename = read_sep(cwd + sep + '/' + folder)
#         sep = pd.DataFrame(dictt)
#         df_diff = calculate_loss(sep, mismart[filename])
#         df_diff.plot()

def read_sep(cwd, sep):
    poz_df_sep = pd.DataFrame()
    neg_df_sep = pd.DataFrame()
    for folder in os.listdir(cwd+sep):
        poz_dict_df = {}
        neg_dict_df = {}
    #     print(folder)
        if folder[0]!='T':
            continue
        else:
            for filename in os.listdir(os.path.join(cwd+sep+'/'+folder)):
        #         if '.csv' not in filename:
        #             print(filename)
                if '86400' in filename:
                    tmp_df = pd.read_csv(os.path.join(cwd+sep+'/'+folder+'/'+filename), sep=";", delimiter=";", index_col=[0], parse_dates=True)
                    tmp_df_poz = tmp_df.loc[tmp_df.VrstaMeritve == "A+_T0_86400_cum_kWh"].Vrednost
                    tmp_df_neg = tmp_df.loc[tmp_df.VrstaMeritve == "A-_T0_86400_cum_kWh"].Vrednost
                    tmp_df_neg = tmp_df_neg.apply(lambda x: float(x.replace(',','.')))
                    tmp_df_neg = tmp_df_neg - tmp_df_neg.shift(periods=1, fill_value=0)
                    tmp_df_poz = tmp_df_poz.apply(lambda x: float(x.replace(',','.')))
                    tmp_df_poz = tmp_df_poz - tmp_df_poz.shift(periods=1, fill_value=0)
                    try:
                        tmp_df_poz[0] = 0
                        tmp_df_neg[0] = 0
                    except:
                        pass
                    poz_dict_df[filename] = tmp_df_poz
                    neg_dict_df[filename] = tmp_df_neg
            dff = pd.DataFrame(poz_dict_df)
            poz_df_sep[folder[0:5]] = dff.sum(axis = 1)
            ddf = pd.DataFrame(neg_dict_df)
            neg_df_sep[folder[0:5]] = ddf.sum(axis = 1)
#     poz_df_sep = poz_df_sep
#     neg_df_sep = neg_df_sep
    return poz_df_sep/60, neg_df_sep/60
        
def get_sum_sep(dict):
    dff = pd.DataFrame(dict)
    dff['suma'] = dff.sum(axis = 1)
    return dff

def read_mismart(directory):
    df_dict = {}
    for filename in os.listdir(directory):
        if '.csv' in filename:
            df_TP = pd.read_csv(directory + '/' + filename, sep="\t", index_col=["Timestamp"], parse_dates=True).resample("D").mean()
            if 'P_W' in df_TP.columns:
                df_dict[filename[:-4]] = (df_TP.P_W)/1000
            else:
                pass
    mismart_df = pd.DataFrame(df_dict)
    mismart_df = mismart_df.dropna(axis=1)
    return mismart_df


def calculate_loss(df1, df2, mutual_tps):
    razlika = []
    finaldf = pd.DataFrame()
    prvadf = pd.DataFrame()
    vtoradf = pd.DataFrame()
    for name in mutual_tps:
        prvadf['Value'] = df1[name].loc[:'2021-03-31 22:00:00+00:00']
        vtoradf['Value'] = df2[name].loc['2019-10-01 22:00:00+00:00':'2021-03-31 22:00:00+00:00']
    #     print(prvadf.shape, vtoradf.shape)
        for i, j in zip(prvadf['Value'], vtoradf['Value']):
            razlika.append(i-j)
        finaldf[name] = razlika
        finaldf[name] = (finaldf[name]/mocnaziv[name])*100
        razlika.clear()
    return finaldf
    
def get_mutual_tps(df_sep, mismart):
    lista = []
    for i in df_sep.columns:
        if i in mismart.columns:
            lista.append(i)
    return lista


def plot_results(finaldf):
    for c in finaldf.columns:
        plt.figure()
        plt.title(f'TP: {c} | Nazivna moc: {mocnaziv[c]}')
    #     plt.xlabel('Timestamp')
        plt.ylabel('Loss in percentage of TP nominal power')
        plt.plot(finaldf[c])
        
def plot_data(dataframe, title = ''):
    plt.figure()
    plt.plot(dataframe)
    plt.title(title)
#     plt.title(ylabel)
    plt.xlabel('Timestamp')
#     plt.ylabel(ylabel)

def generate_anomaly(ts, threshold):
    
#     if(ts1.shape[0]!=ts2.shape[0]):
#         raise ValueError("Time-series must be of same size")
    ts3 = np.empty([ts.shape[0],])
    ts3 = [ts[i] if abs(ts[i])>threshold else np.nan for i in range(ts.shape[0])]
    
    return ts3

def seasonalesd(ts):
    outliers = []
    outliers_indices = sesd.seasonal_esd(ts, hybrid=True, alpha = 3)
    sorted_outliers_indices = np.sort(outliers_indices)
    for idx in sorted_outliers_indices:
        outliers.append(ts[idx])
    marks = [np.nan if i not in outliers else i for i in ts]
    return marks


def load_trtp(path):
    trtp = pd.read_excel(os.path.join(path+'/'+'TR po TP.xlsx'))
    trtp = trtp[['va pa na istem', 'NAZIV_TP', 'TR NAZIVNA MOC']]
    trtp.isna().sum()
    trtp = trtp.dropna()
    trtp['va pa na istem'].astype('int64')
    naziv = [naz[0:5] for naz in trtp.NAZIV_TP]
    nazivna_moc = [moc for moc in trtp['TR NAZIVNA MOC']]
    mocnaziv = dict(zip(naziv,nazivna_moc))
    return mocnaziv