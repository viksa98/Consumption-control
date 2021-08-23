import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    dict_df = {}
    df_sep = pd.DataFrame()
    for folder in os.listdir(cwd+sep):
    #     print(folder)
        if '.DS_Store' in folder:
            pass
        else:
            for filename in os.listdir(os.path.join(cwd+sep+'/'+folder)):
        #         if '.csv' not in filename:
        #             print(filename)
                if '86400' in filename:
                    tmp_df = pd.read_csv(os.path.join(cwd+sep+'/'+folder+'/'+filename), sep=";", delimiter=";", index_col=[0], parse_dates=True)
                    tmp_df_poz = tmp_df.loc[tmp_df.VrstaMeritve == "A+_T0_86400_cum_kWh"].Vrednost
                    tmp_df_poz = tmp_df_poz.apply(lambda x: float(x.replace(',','.')))
                    tmp_df_poz = tmp_df_poz - tmp_df_poz.shift(periods=1, fill_value=0)
                    try:
                        tmp_df_poz[0] = 0
                    except:
                        pass
                    dict_df[filename] = tmp_df_poz
            dff = pd.DataFrame(dict_df)
            df_sep[folder[0:5]] = dff.sum(axis = 1)
        
def get_sum_sep(dict):
    dff = pd.DataFrame(dict)
    dff['suma'] = dff.sum(axis = 1)*4
    return dff

def read_mismart(directory):
    df_dict = {}
    for filename in os.listdir(directory):
        if '.csv' in filename:
            df_TP = pd.read_csv(directory + '/' + filename, sep="\t", index_col=["Timestamp"], parse_dates=True).resample("D").mean()
            if 'P_W' in df_TP.columns:
                df_dict[filename[:-4]] = df_TP.P_W
            else:
                pass
    mismart_df = pd.DataFrame(df_dict)
    mismart_df = mismart.dropna()
    return mismart_df


def calculate_loss(df1, df2, mutual_tps):
    razlika = []
    finaldf = pd.DataFrame()
    prvadf = pd.DataFrame()
    vtoradf = pd.DataFrame()
    for name in lista:
        prvadf['Value'] = df1[name].loc[:'2021-03-31 22:00:00+00:00']
        vtoradf['Value'] = df2[name].loc['2019-10-01 22:00:00+00:00':'2021-03-31 22:00:00+00:00']
    #     print(prvadf.shape, vtoradf.shape)
        for i, j in zip(prvadf['Value'], vtoradf['Value']):
            razlika.append((i-j)/1000)
        finaldf[name] = razlika
        finaldf[name] = finaldf[name]/mocnaziv[name]
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