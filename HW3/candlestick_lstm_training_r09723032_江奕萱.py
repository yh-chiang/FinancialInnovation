from sklearn.metrics import confusion_matrix
import pickle
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam


#讀取資料的部分
def load_data(name):
    # load data from data folder
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

'''
資料前處理 Dataset
'''
def lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes):
    x_train = x_train.reshape(-1, n_step, n_input)
    x_test = x_test.reshape(-1, n_step, n_input)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 #標準化數據
    x_test /= 255 #標準化數據
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return (x_train, x_test, y_train, y_test)


'''
建立LSTM模型:
主要是用一層LSTM以及一層的dense進行預測，
output使用softmax作為計算的fuction

模型調整:
新增Dense(128, activation='relu')
新增Dense(64, activation='relu')
'''
def lstm_model(n_input, n_step, n_hidden, n_classes):
    #添加LSTM、Dense層
    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), unroll=True))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    
    return model

'''
訓練模型:
先設定優化器Adam
再來設定模型的Loss函數、優化器以及用來判斷模型好壞的依據（metrics），此處使用accuracy
最後訓練模型
'''
def lstm_train(model, x_train, y_train, x_test, y_test,learning_rate, training_iters, batch_size):
    adam = Adam(lr=learning_rate)
    model.summary()
    model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train,batch_size=batch_size, epochs=training_iters,verbose=1, validation_data=(x_test, y_test))
'''
辨別機器學習模型的好壞，印出Confusion Matrix:
先得到模型的預測結果
然後與真正的答案做對照
由左上至右下的對角線，代表True Positive和True Negative，
亦即模型預測成功的狀況，因此數字越高越好
'''
def lstm_result(data, x_train, x_test, model):
    # get train & test pred-labels
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)
    # get train & test true-labels
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))
    print(train_result_cm, '\n'*2, test_result_cm)

'''
補充說明:
Batch_Size 設越大，
則跑完一個 Epoch 的時間大約成比例縮小(因為要跑的 Iteration就比例減少)
但設的太大，就必須放大 Epoch 數。
這是因為Batch_Size大，
一個 Epoch 可以跑得 Iteration 數就成比例變少，
就沒有足夠的梯度下降讓損失函數到平穩的低點。
所以必須加大 Epoch 數，這樣訓練時間又變長了，
取捨之間也是必須用觀察的。

參數調整:
learning_rate : 0.001->0.01
batch_size: 128->32
'''
def candlestick_lstm():
    # training parameters
    learning_rate = 0.01 #學習率
    training_iters = 10
    batch_size = 32 # BATCH的大小，相當於一次處理的個數

    # model parameters
    n_input = 40 
    n_step = 10
    n_hidden = 256 #隱含層的特徵數
    n_classes = 10 

    data = load_data('data/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl')
    x_train, y_train, x_test, y_test = data['train_gaf'], data['train_label'][:, 0], data['test_gaf'], data['test_label'][:, 0]
    x_train, x_test, y_train, y_test = lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes)

    model = lstm_model(n_input, n_step, n_hidden, n_classes)
    lstm_train(model, x_train, y_train, x_test, y_test, learning_rate,training_iters, batch_size)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('LSTM test accuracy:', scores[1])
    lstm_result(data, x_train, x_test, model)

if __name__ == '__main__':
    candlestick_lstm()