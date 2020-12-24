# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

#Use LSTM to classify MNIST dataset
'''
#處理資料

MNIST的資料是一個28*28的影像，這裡把他看成一行行的序列（28維度（28長的sequence）*28行）

x 全部都是圖片的資料，
分成 Training data 跟 Test data；
y 全部都是 one-hot encoding 的 Label，
代表著圖片裡的數字是多少，
同樣也是分 Training data 跟 Test data。

標準化數據:
如果要對影像標準化，
因為其中的每一個像素其值都是在0~255之間的一個數字，
所以我們只要把每一個像素的值都除以255就可以讓所有數字被收斂到0~1之間，
完成標準化
'''
# Mnist Dataset
def lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes):
    x_train = x_train.reshape(-1, n_step, n_input) #從一維的784像素轉換成二維的28*28
    x_test = x_test.reshape(-1, n_step, n_input)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #標準化數據
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return (x_train, x_test, y_train, y_test)

'''
建立LSTM模型:
主要是用一層LSTM以及兩層的dense進行預測，
使用softmax作為計算的fuction

模型調整:
dense:增加一層
'''
def lstm_model(n_input, n_step, n_hidden, n_classes):
    #添加LSTM、Dense層
    model = Sequential()
    model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), unroll=True))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model

'''
參數調整:
batch_size:256->128
training_iters = 1->3 
'''

def mnist_lstm_main():
    # training parameters 
    learning_rate = 0.001 #學習率
    training_iters = 3 
    batch_size = 128 # BATCH的大小，相當於一次處理image的個數

    # model parameters  神經網路的參數
    n_input = 28  # x_i 的向量長度，image有28列(輸入一行，一行有28個數據)
    n_step = 28    # 一個LSTM中，輸入序列的長度，image有28行
    n_hidden = 256 #隱含層的特徵數
    n_classes = 10 #輸出的數量，因為是分類問題，0~9個數字，這裡一共有10個

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train_o, y_test_o = lstm_preprocess(x_train, x_test, y_train, y_test, n_step, n_input, n_classes)

    model = lstm_model(n_input, n_step, n_hidden, n_classes)
    trainning(model, x_train, y_train_o, x_test, y_test_o, learning_rate, training_iters, batch_size)
     #驗證模型
    scores = model.evaluate(x_test, y_test_o, verbose=0)
    print('LSTM test accuracy:', scores[1])
    print_confusion_result(x_train, x_test, y_train, y_test, model)
#---------------
    
#Use CNN to classify MNIST dataset
'''
處理資料
x 全部都是圖片的資料，
分成 Training data 跟 Test data；
y 全部都是 one-hot encoding 的 Label，
代表著圖片裡的數字是多少，
同樣也是分 Training data 跟 Test data。
'''
# Mnist Dataset
def cnn_preprocess(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255 #實證表示除以255之後效果會變好
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, x_test, y_train, y_test)

'''
建立cnn模型:
先建立 Convolution 第一層，
然後用 MaxPool 層簡化圖片像素，
重複以上動作一次，
接著用 Flattern 攤平維度，
然後接 Dense 全連接層三層，
最後輸出那10個類別
'''
# Model Structure
def cnn_model():
    # initializing CNN
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(strides=2))
    # Second convolutional layer
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    #將 feature maps 攤平放入一個向量中
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

'''
參數調整:
batch_size:256->128
training_iters = 1->3 
'''
def mnist_cnn_main():
    # training parameters
    learning_rate = 0.001  #學習率
    training_iters = 3 #迭代次數
    batch_size = 64

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test, y_train_o, y_test_o = cnn_preprocess(x_train, x_test, y_train, y_test)

    model = cnn_model()
    trainning(model, x_train, y_train_o, x_test, y_test_o, learning_rate, training_iters, batch_size)
    #驗證模型準確度
    scores = model.evaluate(x_test, y_test_o, verbose=0)
    print('CNN test accuracy:', scores[1])
    print_confusion_result(x_train, x_test, y_train, y_test, model)

#-----------------------------------
    
    
'''
訓練模型:
先設定優化器Adam
再來設定模型的Loss函數、優化器以及用來判斷模型好壞的依據（metrics）
最後訓練模型
'''
def trainning(model, x_train, y_train, x_test, y_test, 
              learning_rate, training_iters, batch_size):
    #設定優化器
    adam = Adam(lr=learning_rate)
    model.summary()
    # 設定模型的Loss函數、優化器以及用來判斷模型好壞的依據（metrics）
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #訓練模型
    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=training_iters,
              verbose=1, validation_data=(x_test, y_test))

'''
辨別機器學習模型的好壞，印出Confusion Matrix：
先得到模型的預測結果
然後與真正的答案做對照
由左上至右下的對角線，代表True Positive和True Negative，
亦即模型預測成功的狀況，因此數字越高越好
'''

def print_confusion_result(x_train, x_test, y_train, y_test, model):
    # get train & test predictions
    train_pred = model.predict_classes(x_train)
    test_pred = model.predict_classes(x_test)
    
    # get train & test true labels
    train_label = y_train
    test_label =  y_test
    
    # confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(10))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(10))
    print(train_result_cm, '\n'*2, test_result_cm)
    

if __name__ == '__main__':
    mnist_lstm_main()
    
    mnist_cnn_main()