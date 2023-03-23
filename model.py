import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import os
import shutil
import time


class Train_N_samples:
    def __init__(self, output_dir, train_data, train_label, valid_data, valid_label, test_data, test_label,
                 sample_length, batch_size, name, epochs=10):
        self.output_dir = output_dir
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.test_data = test_data
        self.test_label = test_label
        self.sample_length = sample_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.name = name

    def document_path(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print('%s文件夹已存在，但是没关系，我们删掉了' % self.output_dir)
        os.mkdir(self.output_dir)
        print('%s已创建' % self.output_dir)

    # 1D-CNN网络结构
    def build_1Dcnn_model(self):
        input_shape = (self.sample_length, 1)
        input = Input(shape=input_shape)
        x = BatchNormalization()(input)
        x = Conv1D(128, 3, activation='selu', kernel_initializer='lecun_normal', padding='same', strides=2)(x)
        x = MaxPool1D(pool_size=(6), strides=2)(x)
        x = Conv1D(64, 3, activation='selu', kernel_initializer='lecun_normal', padding='same', strides=2)(x)
        x = MaxPool1D(pool_size=(4), strides=2)(x)
        x = Conv1D(32, 3, activation='selu', kernel_initializer='lecun_normal', padding='same', strides=2)(x)
        x = MaxPool1D(pool_size=(2), strides=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='selu')(x)
        x = Dense(10, activation='softmax')(x)
        model = Model(input, x)
        return model

    # WDCNN 网络结构，
    def build_WDCNN_model(self):
        input_shape = (self.sample_length, 1)
        input = Input(shape=input_shape)

        # 1
        x = Conv1D(filters=16, kernel_size=64, strides=16, padding='same', activation=tf.nn.relu)(input)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 2
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 3
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 4
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 5
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu)(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)

        x = Flatten()(x)
        x = Dense(100, activation=tf.nn.relu)(x)
        x = BatchNormalization()(x)
        x = Dense(10, activation=tf.nn.softmax)(x)
        model = Model(input, x)
        return model

    # TICNN 网络结构，比wdcnn更深度
    def build_TICNN_model(self):
        input_shape = (self.sample_length, 1)
        input = Input(shape=input_shape)

        # 1
        x = Conv1D(filters=16, kernel_size=8, strides=16, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 2
        x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 3
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 4
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)
        # 5
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)

        # 6
        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size=2, strides=2, padding='valid')(x)

        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(10, activation='softmax')(x)
        model = Model(input, x)
        return model

    # BP 神经网络
    def build_BP_model(self):
        model = Sequential([
            #     BatchNormalization(),
            Dense(units=128, activation=tf.nn.selu, input_shape=[self.sample_length, 1]),
            Dense(units=64, activation=tf.nn.selu),
            Flatten(),
            Dense(units=10, activation=tf.nn.softmax),
        ])
        return model

    # GRU网络结构(Bi)
    def build_GRU_model(self):
        model = Sequential([
            GRU(20, return_sequences=True, activation='relu', input_shape=[self.sample_length, 1]),
            GRU(20, return_sequences=True, activation='relu'),
            Flatten(),
            Dense(10, activation='softmax'),
        ])
        return model

    # LSTM网络结构(Bi)
    def build_LSTM_model(self):
        model = Sequential([
            LSTM(10, return_sequences=True, activation='relu', input_shape=[self.sample_length, 1]),
            LSTM(10, return_sequences=True, activation='relu'),
            Flatten(),
            Dense(10, activation='softmax'),
        ])
        return model

    # RNN网络结构(Bi)
    def build_RNN_model(self):
        model = Sequential([
            SimpleRNN(10, return_sequences=True, activation='relu',
                      input_shape=[self.sample_length, 1]),
            SimpleRNN(10, return_sequences=True, activation='relu'),
            Flatten(),
            Dense(10, activation='softmax'),
        ])
        return model

    """
    =====================网络参数=================
    """

    # 1D_CNN 网络网络训练参数
    def CNN_1D_model(self):
        model = self.build_1Dcnn_model()
        model.summary()
        opt = Adam(lr=0.003)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )

        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since
        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/1DCNN_val_acc.npy', val_acc)
        np.save('./acc/1DCNN_train_acc.npy', train_acc)
        np.save('./acc/1DCNN_val_loss.npy', val_loss)
        np.save('./acc/1DCNN_train_loss.npy', train_loss)
        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])

        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_1DCNN_Time.npy', time_elapsed)
        model_path = self.output_dir + '/1D_CNN' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    # WDCNN 网络网络训练参数
    def WDCNN_model(self):
        model = self.build_WDCNN_model()
        model.summary()
        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )
        # 开始模型训练
        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/WDCNN_val_acc.npy', val_acc)
        np.save('./acc/WDCNN_train_acc.npy', train_acc)
        np.save('./acc/WDCNN_val_loss.npy', val_loss)
        np.save('./acc/WDCNN_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])

        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_WDCNN_Time.npy', time_elapsed)
        model_path = self.output_dir + '/WDCNN' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    # WDCNN_AdaBN 网络网络训练参数
    def WDCNN_AdaBN_model(self):
        model = self.build_WDCNN_model()
        model.summary()
        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )
        # 开始模型训练
        since = time.time()
        model.fit(self.train_data, self.train_label,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  shuffle=True,
                  validation_data=(self.valid_data, self.valid_label)
                  )
        path = './temp/for_wdcnn_AdaBN.h5'
        model.save_weights(path)
        model.load_weights(path)  # 载入模型参数
        model.trainable = False  # 锁定所有层参数
        for i in [2, 5, 8, 11, 14, 18]:  # 只允许特定层更新,BN层对应索引
            model.layers[i].trainable = True
        history = model.fit(self.test_data, self.test_label, epochs=self.epochs, batch_size=self.batch_size)
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/WDCNN_AdaBN_val_acc.npy', val_acc)
        np.save('./acc/WDCNN_AdaBN_train_acc.npy', train_acc)
        np.save('./acc/WDCNN_AdaBN_val_loss.npy', val_loss)
        np.save('./acc/WDCNN_AdaBN_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])

        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_WDCNN_AdaBN_Time.npy', time_elapsed)
        model_path = self.output_dir + '/WDCNN_AdaBN' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    # TICNN 网络网络训练参数
    def TICNN_model(self):
        model = self.build_TICNN_model()
        model.summary()
        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )

        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/TICNN_val_acc.npy', val_acc)
        np.save('./acc/TICNN_train_acc.npy', train_acc)
        np.save('./acc/TICNN_val_loss.npy', val_loss)
        np.save('./acc/TICNN_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])

        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_TICNN_Time.npy', time_elapsed)
        model_path = self.output_dir + '/TICNN' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    def BP_model(self):
        model = self.build_BP_model()
        model.summary()
        opt = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )

        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/BP_val_acc.npy', val_acc)
        np.save('./acc/BP_train_acc.npy', train_acc)
        np.save('./acc/BP_val_loss.npy', val_loss)
        np.save('./acc/BP_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])
        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_BP_Time.npy', time_elapsed)
        model_path = self.output_dir + '/10月06日BP' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    # GRU 网络训练参数
    def GRU_model(self):
        model = self.build_GRU_model()
        model.summary()
        opt = Adam(learning_rate=0.005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )

        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/GRU_val_acc.npy', val_acc)
        np.save('./acc/GRU_train_acc.npy', train_acc)
        np.save('./acc/GRU_val_loss.npy', val_loss)
        np.save('./acc/GRU_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])
        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_GRU_Time.npy', time_elapsed)
        model_path = self.output_dir + '/10月06日GRU' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    def LSTM_model(self):
        model = self.build_LSTM_model()
        model.summary()
        opt = Adam(learning_rate=0.005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )

        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/LSTM_val_acc.npy', val_acc)
        np.save('./acc/LSTM_train_acc.npy', train_acc)
        np.save('./acc/LSTM_val_loss.npy', val_loss)
        np.save('./acc/LSTM_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])
        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_LSTM_Time.npy', time_elapsed)
        model_path = self.output_dir + '/9月23日LSTM' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history

    def RNN_model(self):
        model = self.build_RNN_model()
        model.summary()
        opt = Adam(learning_rate=0.005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'],
                      )

        since = time.time()
        history = model.fit(self.train_data, self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            shuffle=True,
                            validation_data=(self.valid_data, self.valid_label)
                            )
        time_elapsed = time.time() - since

        # 保存epoch过程
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        np.save('./acc/RNN_val_acc.npy', val_acc)
        np.save('./acc/RNN_train_acc.npy', train_acc)
        np.save('./acc/RNN_val_loss.npy', val_loss)
        np.save('./acc/RNN_train_loss.npy', train_loss)

        # 评估模型
        score = model.evaluate(x=self.test_data, y=self.test_label, verbose=0)
        print("测试集上的损失率：", score[0])
        print("测试集上的准确率：", score[1])
        print('The code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))  # 计时
        np.save(self.output_dir + 'Comsuming_RNN_Time.npy', time_elapsed)
        model_path = self.output_dir + '/9月23日RNN' + self.name + '.h5'  # 模型保存
        model.save(model_path)
        print('%s已保存' % model_path)
        return history
