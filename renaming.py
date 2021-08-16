import math
import os
import pandas as pd


def renaming(path_trtp):

    trtp = pd.read_excel(os.path.join(path_trtp+'/'+'TR po TP.xlsx'))
    trtp = trtp[['va pa na istem', 'NAZIV_TP']]
    trtp.isna().sum()

    trtp = trtp.dropna()
    trtp['va pa na istem'].astype('int64')

    cwd = os.getcwd()
    sifri = [math.trunc(sifra) for sifra in trtp['va pa na istem']]
    sifri = list(map(str, sifri))
    desc_name = [name[0:5] for name in trtp.NAZIV_TP]
    new_dict = dict(zip(sifri, desc_name))
    for file in os.listdir(cwd+'/Mismart'):
        if '.csv' in file:
            f_name, f_ext = os.path.splitext(file)
            if file[0:7] in new_dict.keys():
                f_name = new_dict[f_name]
                new_name = '{}{}'.format(f_name, f_ext)
                os.rename(os.path.join(cwd+'/Mismart'+'/'+file), os.path.join(cwd+'/Mismart'+'/'+new_name))
            else:
                pass
        else:
            pass