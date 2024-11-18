import pandas as pd
import numpy as np
def Seq_input():
    text_list = []
    with open("E:\数据\最终数据\序列数据\mRNA\mRNASequence.txt", 'r') as file:
        for line in file:
            text = line.split(":")[-1].strip()
            # text_list.append(line.scrip())
            text_list.append(text)
    return text_list

def Exp_input():
    data = pd.read_excel(r"E:\数据\Cancer_data\PRAD\PRAD_correct\PRAD_miRNA_exp.xlsx")
    return data


def Model_Data_Name():
    Name = pd.read_excel(r"E:\数据\Cancer_data\BRCA\BRCA_Vec\BRCA_Train_name.xlsx")
    Name=pd.DataFrame(Name)
    return Name


def Mode_Data_Embedding():
    combined_embeddings = pd.read_excel("E:\数据\Cancer_data\BRCA\BRCA_Vec\BRCA_Train_data.xlsx")
    combined_embeddings = pd.DataFrame(combined_embeddings)
    return combined_embeddings


def Mode_Data_Label():
    label=pd.read_excel(r"E:\数据\Cancer_data\BRCA\BRCA_Vec\BRCA_Train_lables.xlsx")
    label = pd.DataFrame(label)
    return label