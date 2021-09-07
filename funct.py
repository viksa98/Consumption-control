import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import sesd

def read_sep(abs_path_sep, pickle_filename, abs_output_pickle_path):
    
    """
    Function that reads the SEP data per TPs into a pandas DataFrame
    
    Parameters:
    
    abs_path_sep -> input path of the SEP data
    
    pickle_filename -> name of the pickle file

    abs_output_pickle_path -> output path of the pickle file
    
    Output:
    
    pandas DataFrame containing Sep data per TPs, function also creates a pickle file that contains the SEP data
    
    """
    
    poz_df_sep = pd.DataFrame()
    neg_df_sep = pd.DataFrame()
    for folder in os.listdir(abs_path_sep):
        poz_dict_df = {}
        neg_dict_df = {}
        if folder[0]!='T':
            continue
        else:
            for filename in os.listdir(abs_path_sep+'/'+folder):
                if '86400' in filename:
                    tmp_df = pd.read_csv(os.path.join(abs_path_sep+'/'+folder+'/'+filename), sep=";", index_col=[0], parse_dates=True)
                    tmp_df_poz = tmp_df.loc[tmp_df.VrstaMeritve == "A+_T0_86400_cum_kWh"].Vrednost
                    tmp_df_poz = tmp_df_poz.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x).resample("D").mean().diff(periods=1).fillna(0)
                    tmp_df_neg = tmp_df.loc[tmp_df.VrstaMeritve == "A-_T0_86400_cum_kWh"].Vrednost
                    tmp_df_neg = tmp_df_neg.apply(lambda x: float(x.replace(',','.')) if isinstance(x,str) else x).resample("D").mean().diff(periods=1).fillna(0)
                    poz_dict_df[filename] = tmp_df_poz
                    neg_dict_df[filename] = tmp_df_neg
            poz_df_sep[folder[0:5]] = pd.DataFrame(poz_dict_df).sum(axis = 1)
            neg_df_sep[folder[0:5]] = pd.DataFrame(neg_dict_df).sum(axis = 1)
            sep_data_tps = poz_df_sep.sub(neg_df_sep)
    sep_data_tps*=0.04166667
    sep_data_tps.to_pickle(abs_output_pickle_path+'/'+pickle_filename)
    return sep_data_tps

def read_mismart(abs_path_mismart, pickle_filename, abs_output_pickle_path):
    
    """
    Function that reads the Mismart data per TPs into a pandas DataFrame
    
    Parameters:
    
    abs_path_mismart -> input path of the Mismart data
    
    pickle_filename -> name of the pickle file

    abs_output_pickle_path -> output path of the pickle file
    
    Output:
    
    pandas DataFrame containing Mismart data per TPs, function also creates a pickle file that contains the Mismart data.
    
    """
    
    df_dict = pd.DataFrame()
    for filename in os.listdir(abs_path_mismart):
        if '.csv' in filename:
            df_TP = pd.read_csv(abs_path_mismart + '/' + filename, sep="\t", index_col=["Timestamp"], parse_dates=True).resample("D").mean()
            if 'P_W' in df_TP.columns:
                df_dict[filename[:-4]] = (df_TP.P_W)/1000
            else:
                pass
    df_dict.dropna(axis=1, inplace=True)
    df_dict.to_pickle(abs_output_pickle_path + "/" + pickle_filename)
    return df_dict


def calculate_loss(mismart, sep, mutual_tps, nominal_power, start_date, end_date):
    
    """
    Function that calculates the overall losses from Mismart to SEP measuring instruments
    
    Parameters:
    
    mismart -> pandas DataFrame containing values for the losses for the mutual TPs over the whole time interval of interest
    
    sep -> pandas DataFrame containing Mismart data per TPs
    
    mutual_tps -> python list with the mutual TPs
    
    nominal_power -> python dictionary containing the names and nominal power for all the TPs
    
    start_date -> starting point for the calculations in the following format: yyy-mm-dd hh:mm:ss
    
    end_date -> starting point for the calculations in the following format: yyy-mm-dd hh:mm:ss
    
    Output:
    
    pandas DataFrame with overall losses for every TPs for every timestamp
    
    """
    
    finaldf = pd.DataFrame()
    dict1 = {}
    dict2 = {}
    for name in mutual_tps:
        dict1['Value'] = mismart[name].loc[start_date:end_date]
        dict2['Value'] = sep[name].loc[start_date:end_date]
        finaldf[name] = [(i-j) for i,j in zip(dict1['Value'], dict2['Value'])]
        finaldf[name] = (finaldf[name]/nominal_power[name])*100
    return finaldf
    
def get_mutual_tps(sep_df, mismart_df):
    
    """
    Function that returns a list of TPs whose data is contained in both Mismart and SEP measurements
    
    Parameters:
    
    sep_df -> pandas DataFrame containing SEP data per TPs (read_sep(**kwargs) output)
    
    mismart_df -> pandas DataFrame containing Mismart data per TPs (read_mismart(**kwargs) output)
    
    Output:
    
    python list with the mutual TPs
    
    """
    
    return [i for i in sep_df.columns if i in mismart_df.columns]


def plot_results(loss_data, nominal_power):
    
    """
    The function plots visual graphs of the overall losses (in percentage of TP nominal power) between Mismart and SEP measurement instruments
    
    Parameters:
    
    loss_data -> pandas DataFrame containing values for the losses for the mutual TPs over the whole time interval of interest
    
    nominal_power -> python dictionary containing the names and nominal power for all the TPs
    
    """
        
    for c in loss_data.columns:
        l1 = generate_anomaly(loss_data[c].to_numpy(), 0.05*nazivna_moc[c])
        plt.figure()
        plt.title(f'TP: {c} | Nazivna moc: {nominal_power[c]}')
        plt.ylabel('Loss in percentage of TP nominal power')
        plt.plot(loss_data[c], linestyle = '-', label ='Loss curve')
        plt.plot(l1, 'ro', markersize = "3", label ='Anomalies')
        plt.legend()

def generate_anomaly(ts, threshold):
    
    """
    Function that returns a list containing the anomalies of a timeseries
    
    Parameters:
    
    ts -> timeseries in a numpy.array format
    
    threshold -> numeric value such that any instance of the timeseries which is greater than this value is considered anomalous
    
    Output:
    
    Python list containing anomalous values of the input timeseries. The list is of a same shape as the input, having np.nan instance if the value is not an anomaly, and otherwise the actual value
    
    """
    
    return [ts[i] if abs(ts[i])>threshold else np.nan for i in range(ts.shape[0])]

def seasonalesd(ts):
    
    """
    Function that returns a list containing the anomalies of a timeseries calculated with respect to SESD algorithm.
    Installation of sesd package is a requirement.
    
    Parameters:
    
    ts -> timeseries in a numpy.array format
    
    Output:
    
    Python list containing anomalous values of the input timeseries. The list is of a same shape as the input, having np.nan instance if the value is not an anomaly, and otherwise the actual value
    
    """
    
    outliers_indices = np.sort(sesd.seasonal_esd(ts, hybrid=True, alpha = 3))
    outliers = [ts[idx] for idx in outliers_indices]
    return [np.nan if i not in outliers else i for i in ts]


def load_trtp(path):
    
    """
    Function that returns a dictionary containing the names and nominal power for all the TPs
    
    Parameters:
    
    path -> path to the directory where the Excel file with the data is located
    
    Output:
    
    python dictionary containing the names and nominal power for all the TPs
    
    """
    
    trtp = pd.read_excel(os.path.join(path+'/'+'TR po TP.xlsx'))
    trtp = trtp[['va pa na istem', 'NAZIV_TP', 'TR NAZIVNA MOC']].dropna()
    trtp['va pa na istem'].astype('int64')
    naziv = [naz[0:5] for naz in trtp.NAZIV_TP]
    nazivna_moc = [moc for moc in trtp['TR NAZIVNA MOC']]
    return dict(zip(naziv,nazivna_moc))

def load_pickle_df(filename):
    
    """
    Function that that loads a pickle file in a pandas DataFrame.
    
    Parameters:
    
    filename -> name of the pickle file
    
    Output:
    
    pandas DataFrame containing the content of the pickle file
    
    """
    
    cwd = os.getcwd()
    with open(os.path.join(cwd, filename), 'rb') as handle:
        b = pickle.load(handle)
    return pd.DataFrame(b)

def plot_sep_mismart(sep_data, mismart_data, mutual_tps):
    
    """
    Function that plots in a subplot SEP and Mismart timeseries for all the TPs
    
    Parameters:
    
    sep_data -> pandas DataFrame containing SEP data per TPs
    
    mismart_data -> pandas DataFrame containing Mismart data per TPs
    
    mutual_tps -> python list with the mutual TPs
    
    """
    
    plt.figure(figsize=(32, 64))
    j = 1
    x = 2
    for i in range(len(mutual_tps)):
        plt.subplot(len(mutual_tps), 2, j)
        plt.title(f'SEP data for {mutual_tps[i]}')
        plt.plot(sep_data[mutual_tps[i]])
        j+=2
    for i in range(len(mutual_tps)):
        plt.subplot(len(mutual_tps), 2, x)
        plt.title(f'Mismart data for {mutual_tps[i]}')
        plt.plot(mismart_data[mutual_tps[i]])
        x+=2

if __name__ == "__main__":
    #cwd = os.getcwd()
    #sep = 'Podatki SEP2'
    ##sep_data_tps = read_sep(cwd, sep, 'sep_pkl.pkl')
    mismart_data = read_mismart('C:\\Users\\bldob\\Desktop\\consumption-control-github\\Consumption-control\\Mismart', 'mismart_pkl.pkl', 'C:\\Users\\bldob\\Desktop\\consumption-control-github\\Consumption-control')
    print(mismart_data)
    #mutual_tps = get_mutual_tps(sep_data_tps, mismart_data)
    #nazivna_moc = load_trtp('../Podatki')
    #loss_data = calculate_loss(mismart_data, sep_data_tps, mutual_tps, nazivna_moc, '2019-10-01 22:00:00+00:00', '2021-03-31 22:00:00+00:00')
    #print(nazivna_moc)
    #print(loss_data.to_numpy())
    #import shutil
    #import os
    #print(os.getcwd())
    #functionality for removing non-relevant TPs

    #for root,folder,files in os.walk("C://Users//bldob//Desktop//consumption-control-github//Consumption-control//Mismart"):
    #    for f in files:
    #        if ".csv" in f:
    #            new_path = root.split("\\")[0]
    #            #print(new_path)
    #            print(root + "//" + f)
    #            print(new_path + "//" + f)
    #            shutil.move(root + "//" + f, new_path + "//" + f)