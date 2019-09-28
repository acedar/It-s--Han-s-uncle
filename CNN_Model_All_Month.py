import os
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import models
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D
import Training_Figure
import New_Predicted__Indian_Data
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model
model=models.Sequential()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"# 使用GPU运行,CUDA_VISIBLE_DEVICES=-1使用cpu运行
np.set_printoptions(suppress=True)# numpy矩阵数据精度不显示

seed = 7
np.random.seed(seed)

# 特征总数，划分出特征值和预测值出来
feature_N = 1875
# 目标位置
path = "E:/pycharm/charm/untitled/ocean_data/CNN_Indian_Ocean"


def data_pre_processing(year,month):
    # pandas读取数据集
    data = pd.read_csv(path + '/data/' + year + '.csv', header=None)
    shape_Data=data.shape
    print("The shape of original_data is："+str(shape_Data))#数据预处理
    data.dropna(inplace=True) # pandas删除含有缺失值（NAN表示）的行，不包括99999表示的缺失值,numpy没有这个函数

    # shape = data.shape
    # print("原数据：" + str(shape))
    # for x in range(0, shape[0]):
    #     for y in range(1879, shape[1]):
    #         if data.iloc[x, y] <= -1000:
    #             data.iloc[x, y] = None
    # data.dropna(inplace=True)
    # print("后数据：" + str(data.shape))

    #pandas临时删除不可用的特征值：年、月、经、纬
    temp_col_position=[0,1,2]
    #pandas删除指定的某几列数据
    # for col in range(6,1879,3):
    #     temp_col_position.append(col)
    # data.drop(data.columns[temp_col_position],axis=1,inplace=True)
    # temp_col_position=[1]
    data.drop(data.columns[temp_col_position],axis=1,inplace=True)
    All_M =data.shape[0]#样本总数，m=data.shape[:,0:]表示多少特征

    print("The shape of new_data is：" + str(data.shape))
    print("I have m=%d data set!" % (All_M))



    data=data.values#将DataFrame格式数据转换成numpy矩阵形式,处理数据更快
    np.random.shuffle(data)# 打乱数据集,防止特数据集特殊规律的影响

    X_training_test = data[:, :feature_N]#提取X特征数据：csv中0-1874列
    print(X_training_test[0:3, 0:10])#查看特征前几行
    print(X_training_test[0:3, feature_N-2:feature_N])#查看提取的X是否准确，防止把y值提取处理

    data_X_training_test=preprocessing.scale(X_training_test)
    scaler=preprocessing.StandardScaler().fit(X_training_test)
    print("10年特征平均值及标准差：")
    scaler_mean=scaler.mean_.reshape(1,feature_N)
    scaler_std=np.sqrt(scaler.var_.reshape(1,feature_N))#StandardScaler()中不包含方差，只包含标准差（标准差为方差的取根号）
    scaler_mean_std = np.hstack((scaler_mean,scaler_std))#拼接均值和方差为同一二维矩阵
    scaler_mean_std = scaler_mean_std.reshape(2, feature_N)
    scaler_mean_std = pd.DataFrame(scaler_mean_std)
    print(scaler_mean_std)
    scaler_mean_std.to_csv(path+'/data/scaler_mean_std_month_'+str(month)+'.csv', header=True, index=False)
    #经过标准化后：训练集和测试集上特征数据均值为0，方差为1
    print("标准化后：")
    print(data_X_training_test.mean(axis=0))
    print(data_X_training_test.std(axis=0))

    Trainging_M=int(0.98*All_M)#留出2%的数据作为测试集
    Test_M=All_M-Trainging_M-2
    print("训练样本数："+str(Trainging_M))
    print("测试样本数"+str(Test_M))

    #训练集和测试集X提取
    X_training=data_X_training_test[:Trainging_M,:]#取1878个features，43000样本作为训练集(10%为验证集)
    print(X_training[0:3,0:3])
    print(X_training.shape)
    y_training = data[:Trainging_M, feature_N:]#训练集y值提取
    print(y_training[0:3,0:3])#查看提取的是否符合
    print(y_training.shape)#查看输出值y的维度
    Layer_M=y_training.shape[1]#一共多少输出层
    print(Layer_M)
    print(y_training)
    #
    # y_training-=y_training.mean(axis=1).reshape(Trainging_M,1)#除去行的均值（当前样本所有层的温度均值--如何解释？为何要这样？）

    # y_training-= y_training.mean(axis=0)#1.除去列的平均值（全部样本当前层的均值---如何解释？）


    X_test=data_X_training_test[Trainging_M:All_M,:]#取测试特征数据
    y_test = data[Trainging_M:All_M, feature_N:]#取测试真实值

    # y_test-=y_test.mean(axis=1).reshape(All_M-Trainging_M,1)#除去行的均值
    # y_test-= y_test.mean(axis=0)#1.除去列的平均值

    #返回：训练集X,Y,测试集X，Y，训练集总数，全部样本数，测试集总数
    del data
    return X_training,y_training,X_test,y_test,Trainging_M,All_M,Test_M,Layer_M
def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred - K.mean(y_true)), axis=-1)
    SST = K.mean(K.square(y_true - K.mean(y_true)), axis=-1)
    return SSR / SST


def score(y_true, y_pred):
    score = r2_score(y_true, y_pred, multioutput='raw_values')
    print(score)
def set_model_AlexNet(X_training, Y_training,Layer_M,month):
    """
    模型AlexNet
    
    :param X_training: 
    :param Y_training: 
    :return: 
    """
    model.add(Conv2D(96, (3, 2), strides=(3, 1), input_shape=(57, 33, 1), padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(Layer_M))
    model.compile(optimizer='Adam', loss='mse', metrics=['mae', 'mse',r_square])
    print(model.summary())
    print("lr {}".format(K.get_value(model.optimizer.lr)))
    plot_model(model, show_shapes=True,
               to_file=path + '/result/new_result/' + str(month) + '/EININO_CNN_model_month_' + str(
                   month) + '.png')
    history=model.fit(X_training, Y_training, validation_split=0.2, batch_size=256, epochs=100, verbose=1)
    model.save(path+'/result/new_result/'+str(month)+'/my_CNN_model_month_'+str(month)+'.h5')
    print("success save the CNN_model_STA")
    Training_Figure.figuer_value(history,month,path)
    print("绘制训练和验证loss、MAE、MSE成功")



def predicted_data(X_test,y_test,Test_M,Layer_M,month):
    """
    测试数据集：评估，绘图，曲线对比
    :param X_test: 
    :param y_test: 
    :return: 
    """
    # 300个数据进行评估，返回全部样本的损失值(loss=mse)和平均绝对误差(mae),均方误差（mse）
    loss=model.evaluate(X_test,y_test)
    print(loss)
    #如何输出各层的mse以及r_squared?也就是会有52个mse和r_squared

    #0-1975m之间的57（层）个点预测值和真实值对比
    depth = [5,10,20,30,40,50,60,70,80,90,
            100,110,120,130,140,150,160,170,180,200,
            220,240,260,280,300,320,340,360,380,400,420,440,460,500,
            550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,
             1300,1400,1500,1600,1700,1800,1900,1975]


    #使用50个随机数据来观测拟合程度,并绘制预测值和真实值对比图像
    plt.clf()
    Test_M=int(float(Test_M)/30)
    for point in range(0,30):
        test_point=X_test[point*Test_M:point*Test_M+1,:]# 从测试集3000个点钟每隔45个间隔取一个点用作测试
        model_predicted_value = model.predict(test_point)#逐个点通过模型预测，返回1*35维度的矩阵
        # 将单个样本预测结果放在列表中
        predicted_value=[]
        for index in range(0,Layer_M):
            predicted_value.append(model_predicted_value[0,index])
        print("figure--:"+str(point))
        # 拟合x和y的数据集为一条平滑曲线：红色线为预测值
        # poly = np.polyfit(x, y, deg=10)
        # z = np.polyval(poly, x)
        plt.plot(predicted_value, depth, c="red",marker='o')
        # 真实值取和预测值相同行位置的值存入列表中
        real_value = []
        for layer in range(0,Layer_M):
            real_value.append(y_test[point*Test_M,layer])
        plt.plot(real_value, depth, c="blue",marker='x')
        #反转xy轴
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        #设置图像样式
        plt.xlabel('Temperature (℃)')
        plt.ylabel('Depth (m)')
        # plt.title('real and predicted value')
        MSE = mean_squared_error(real_value, predicted_value)
        R2 = r2_score(real_value, predicted_value)
        real_value.sort()
        plt.text(real_value[1], 400, 'MSE=' + str(round(MSE, 3)))
        plt.legend()#真实值和预测值图像绘制在同一坐标中
        plt.grid(linestyle='--')
        plt.savefig(path+'/result/new_result/'+str(month)+'/'+str(point+1) + '.png')# 保存图像
        plt.cla()# Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
if __name__=='__main__':
    All_Year=['200510_201410','200504_201404','200502_201402','200503_201403','200505_201405','200506_201406','200508_201408','200509_201409','200511_201411','200512_201412',]
    All_Month=['10','3','4','3','5','6','8','9','11','12']
    All_Test_Year=['201510','201502','200404','201504']
    for i in range(0,1):
        year=All_Year[i]
        month=All_Month[i]
        print(year)
        print(month)
        X_training, Y_training,X_test,y_test,Trainging_M,All_M,Test_M,Layer_M=data_pre_processing(year,month)
        # 为了能够符合CNN的要求，必须给数据集加上6列0值，能够拼凑出图像类型57*33
        X_training_zero = np.zeros([Trainging_M, 6])
        X_test_zero = np.zeros([All_M - Trainging_M, 6])
        X_training = np.c_[X_training, X_training_zero]
        X_training=X_training.reshape(Trainging_M,57,33,1)

        X_test = np.c_[X_test, X_test_zero]
        X_test = X_test.reshape(All_M-Trainging_M, 57, 33, 1)

        set_model_AlexNet(X_training, Y_training,Layer_M,month)#使用AlexNet模型
        predicted_data(X_test,y_test,Test_M,Layer_M,month)#测试集上测试模型并绘制图像


        # print("开始2004及2015的预测:")
        # for j in range(0,1):
        #     New_Predicted__Indian_Data.data_pre_processing(All_Test_Year[i+j],All_Month[i],path)

        # 释放内存，防止Merry error
        # del X_training, Y_training, X_test, y_test, Trainging_M
        # gc.collect()