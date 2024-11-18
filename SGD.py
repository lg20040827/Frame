import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import InPut
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def process_data():
    combined_embeddings=InPut.Mode_Data_Embedding()
    Name=InPut.Model_Data_Name()
    len1=len(combined_embeddings)-1
    len2=int(len1*0.5)
    combined_embeddings=combined_embeddings.iloc[1:, 1:].values[:len1]
    combined_embeddings=np.array(combined_embeddings)
    # Create a Min-Max Normalizer
    scaler = MinMaxScaler()
    # Perform Min-Max Normalization on the Data
    X = scaler.fit_transform(combined_embeddings)
    # Read Excel File
    y = InPut.Mode_Data_Label()
    #Shuffling Operation
    y = y.iloc[0:,1:].values[:len1]
    Name=Name.iloc[0:,1:].values[:len1]
    np.random.seed(42)
    shuffle_index = np.random.permutation(len1)
    X,y,Name = X[shuffle_index],y[shuffle_index],Name[shuffle_index]
    y = y.reshape(-1)
    X_train,X_text,y_train,y_text,Name_train,Name_text= X[:len2],X[len2:],y[:len2],y[len2:],Name[:len2],Name[len2:]
    y_train=(y_train==1)
    return X_train,X_text,y_train,y_text,Name_train,Name_text


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(
    max_iter=1000,
    random_state=42,
    eta0=0.00001,
    learning_rate='constant',
    alpha=0.0001,
    penalty='elasticnet'
)
from sklearn.model_selection import StratifiedGroupKFold
skflods = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)


def cross_val(X_train,y_train):
    from sklearn.model_selection import cross_val_score
    cross_val_score(sgd_clf,X_train,y_train,cv=5,scoring='accuracy')
    print(cross_val_score(sgd_clf,X_train,y_train,cv=5,scoring='accuracy'))
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict
    y_train_pred = cross_val_predict(sgd_clf,X_train,y_train,cv=5)
    print(confusion_matrix(y_train,y_train_pred))
    # y_scores =  cross_val_predict(sgd_clf,X_train,y_train,cv=5,method="decision_function")


def train(X_train,y_train,X_text):
    sgd_clf.fit(X_train,y_train)
    y_pred = sgd_clf.predict(X_text)
    y_text_pred= sgd_clf.decision_function(X_text)
    return y_pred,y_text_pred

def Get_SGD_AUC():
    X_train, X_text, y_train, y_text, Name_train, Name_text = process_data()
    cross_val(X_train, y_train)
    y_pred,y_text_pred=train(X_train, y_train,X_text)
    fpr, tpr, thresholds = roc_curve(y_text, y_text_pred, drop_intermediate=False)
    auc = roc_auc_score(y_text, y_text_pred)
    return fpr,tpr,auc

def Get_SGD_PUC():
    X_train, X_text, y_train, y_text, Name_train, Name_text = process_data()
    y_pred, y_text_pred = train(X_train, y_train,X_text)
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_text, y_text_pred)
    pr_auc = auc(recall, precision)
    return precision, recall , pr_auc
