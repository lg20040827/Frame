import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import Doc2Vec
import Role2Vec
import LSTM
import SGD

def process_data():
    Exp_Vec=Role2Vec.get_expvec()
    print(Exp_Vec[1])
    Seq_Vec=Doc2Vec.get_seqvec()
    print(Seq_Vec[1])
    df_seq=pd.DataFrame(Seq_Vec)
    df_exp=pd.DataFrame(Exp_Vec)
    # df_seq.to_excel("E:\data\finall\seq\mSequenceVec.xlsx")
    # df_exp.to_excel("E:\data\cancer\PRAD\PRAD_Vec\PRAD_miRNA_ExpVec.xlsx")