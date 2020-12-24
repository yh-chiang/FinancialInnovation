from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPool2D


#讀取資料
def load_data(name):
    # load data from data folder
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

'''
建立CNN模型：
先建立了 Convolution 層(兩層)，
接著用 Flattern 攤平維度，
然後接 Dense 全連接層三層，
最後輸出9個類別
'''
# Model Structure
def cnn_model(params):
    # initializing CNN
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(10, 10, 4)))
    # Second convolutional layer
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu')) 
    #將 feature maps 攤平放入一個向量中
    model.add(Flatten()) #攤平維度
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    return model


    
'''
訓練模型：
選擇使用CNN Model
設定模型的Loss函數、優化器以及用來判斷模型好壞的依據（metrics），這裡用準確度來衡量
最後訓練模型
'''
def cnn_train(params, data):
    model = cnn_model(params)
    # 設定模型的Loss函數、優化器以及用來判斷模型好壞的依據（metrics）
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    #訓練模型
    hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    return (model, hist)

'''
辨別機器學習模型的好壞，印出Confusion Matrix：
先得到模型的預測結果
然後與真正的答案做對照
由左上至右下的對角線，代表True Positive和True Negative，
亦即模型預測成功的狀況，因此數字越高越好
'''

def cnn_result(data, model):
    # get train & test pred-labels
    train_pred = model.predict_classes(data['train_gaf'])
    test_pred = model.predict_classes(data['test_gaf'])
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
一個 Epoch 可以跑得 Iteration 數就成比例變少，就
沒有足夠的梯度下降讓損失函數到平穩的低點。
所以必須加大 Epoch 數，這樣訓練時間又變長了，
取捨之間也是必須用觀察的。

參數調整:
learning_rate : 0.01->0.05
batch_size: 64->32
'''
def candlestick_cnn():
    PARAMS = {}
    PARAMS['pkl_name'] = 'data/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl' 
    # training parameters 
    PARAMS['classes'] = 9 
    PARAMS['lr'] = 0.05  #學習率
    PARAMS['epochs'] = 10  #決定訓練要跑幾回合 epoch，一個 epoch 就是全部的訓練數據都被掃過一遍。
    PARAMS['batch_size'] = 32 # BATCH的大小，相當於一次處理的個數
    PARAMS['optimizer'] = optimizers.SGD(lr=PARAMS['lr'])
    # load data & keras model
    data = load_data(PARAMS['pkl_name'])
    # train cnn model
    model, hist = cnn_train(PARAMS, data)
    # 驗證模型
    scores = model.evaluate(data['test_gaf'], data['test_label_arr'], verbose=0)
    print('CNN test accuracy:', scores[1])
    cnn_result(data, model)
    
if __name__ == "__main__":
    candlestick_cnn()