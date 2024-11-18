import pandas as pd
import numpy as np
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model as KerasModel
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import InPut


def process_data():
    combined_embeddings = InPut.Mode_Data_Embedding()
    len1 = len(combined_embeddings) - 1
    len2 = int(len1 * 0.5)
    combined_embeddings = combined_embeddings.iloc[1:, 1:].values[:len1]
    Name = InPut.Model_Data_Name()
    lab = InPut.Mode_Data_Label()
    values_to_assign = lab.iloc[:, 1:].values[:len1]
    Name = Name.iloc[:, 1:].values[:len1]

    np.random.seed(42)
    shuffle_index = np.random.permutation(len1)
    values_to_assign, combined_embeddings, Name = values_to_assign[shuffle_index], combined_embeddings[shuffle_index],Name[shuffle_index]

    values_to_assign, y_text, combined_embeddings, x_text, Name, z_text = values_to_assign[:len2], values_to_assign[len2:], combined_embeddings[:len2], combined_embeddings[len2:], Name[:len2], Name[len2:]

    combined_embeddings = np.array(combined_embeddings)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    combined_embeddings = scaler.fit_transform(combined_embeddings)

    # Data Reshaping
    combined_embeddings_reshaped = combined_embeddings.reshape((len2, 1, 768))
    x_test = x_text.reshape(len1 - len2, 1, 768)

    labels_train = values_to_assign
    labels_train = labels_train.reshape((len2, 1, 1))

    lables_test = y_text
    y_test = lables_test.reshape((len1 - len2, 1, 1))

    X = combined_embeddings_reshaped
    y = labels_train
    return X, y, x_test, y_test


def LSTM_Frame(X):
    input_data = Input(shape=(X.shape[1], X.shape[2]), name='input_data')

    cs = 0.0001
    lstm_output_1 = LSTM(units=32, activation='tanh', return_sequences=True,
                         kernel_regularizer=regularizers.l2(cs),
                         recurrent_regularizer=regularizers.l2(cs),
                         bias_regularizer=regularizers.l2(cs))(input_data)
    lstm_output_1 = Dropout(0.01)(lstm_output_1)

    lstm_output_2 = LSTM(units=32, activation='tanh', return_sequences=True,
                         kernel_regularizer=regularizers.l2(cs),
                         recurrent_regularizer=regularizers.l2(cs),
                         bias_regularizer=regularizers.l2(cs))(lstm_output_1)
    lstm_output_2 = Dropout(0.01)(lstm_output_2)

    output = Dense(units=1, activation='sigmoid')(lstm_output_2)
    return input_data, output


def train_model(input_data, output, X, y, x_test, y_test):
    model = KerasModel(inputs=input_data, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-05), metrics=['accuracy'])
    model.fit(X, y, batch_size=30, epochs=800, validation_data=(x_test, y_test))
    y_test_prob = model.predict(x_test)
    return y_test_prob


def Get_LSTM_AUC():
    X, y, x_test, y_test = process_data()
    input_data, output = LSTM_Frame(X)
    y_test_prob = train_model(input_data, output, X, y, x_test, y_test)

    fpr, tpr, thresholds = roc_curve(y_test.flatten(), y_test_prob.flatten())
    auc_score = roc_auc_score(y_test.flatten(), y_test_prob.flatten())
    return fpr, tpr, auc_score


def Get_LSTM_PRC():
    X, y, x_test, y_test = process_data()
    input_data, output = LSTM_Frame(X)
    y_test_prob = train_model(input_data, output, X, y, x_test, y_test)

    precision, recall, _ = precision_recall_curve(y_test.flatten(), y_test_prob.flatten())
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

if __name__ == "__main__":

    fpr, tpr, auc_score = Get_LSTM_AUC()
    print(f"AUC: {auc_score}")


    precision, recall, pr_auc = Get_LSTM_PRC()
    print(f"PR AUC: {pr_auc}")
