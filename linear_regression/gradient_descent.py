import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


attrs = ['AMB', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
iterator = 100
lr_rate = 0.01

def read_TrainData(filename, N):
    #N: how many hours to be as inputs
    raw_data = pd.read_csv(filename).as_matrix()
    # 12 months, 20 days per month, 18 features per day. shape: (4320 , 24)
    data = raw_data[:, 3:] #first 3 columns are not data
    data = data.astype('float')
    X, Y = [], []
    for i in range(0, data.shape[0], 18*20):
        # i: start of each month
        days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
        concat = np.concatenate(days, axis=1) # shape: (18 feat, 480(day*hr))
        # take every N hours as x and N+1 hour as y
        for j in range(0, concat.shape[1]-N):
            features = concat[:, j:j+N].flatten() #the data of previous N hours
            features = np.append(features, [1]) # add w0
            X.append(features)
            Y.append([concat[9, j+N]]) #9th feature is PM2.5
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

#from 1/23 0am, 1am ..23pm... 2/23, 0am, .... ~ 12/31 23p.m, total 2424 hours
#will give you a matrix 2424 * (18*N features you need)
def read_TestData(filename, N):
	#only handle N <= 48(2 days)
    assert N <= 48
    raw_data = pd.read_csv(filename).as_matrix()
    data = raw_data[:, 3:]
    data = data.astype('float')
    surplus = DAYS - 20 #remaining days in each month after 20th
    test_X = []
    test_Y = [] #ground truth
    for i in range(12): # 12 month
        # i: start of each month
        start = sum(surplus[:i])*18
        end = sum(surplus[:i+1])*18
        days = np.vsplit(data[start:end], surplus[i])
        concat = np.concatenate(days, axis=1) # shape: (18 feat, (day*hr))
        for j in range(48, concat.shape[1]): #every month starts from 23th
            features = concat[:, j-N:j].flatten()
            features = np.append(features, [1]) # add w0
            test_X.append(features)
            test_Y.append([concat[9, j]])
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    return test_X, test_Y


class Linear_Regression(object):
    def __init__(self ):
        self.W = np.full( (1 , N*18+1) , 0.1 )
    def train(self, train_X, train_Y):
        #TODO
        #W = ?
        '''
        ##Linear_Regression
        W = np.matmul(np.linalg.inv(np.matmul(np.array(train_X).transpose() , np.array(train_X))) , np.array(train_X).transpose())
        W = np.matmul( W ,train_Y )
        self.W = W #save W for later prediction
        predict_Y = self.predict( train_X )
        loss = MSE( predict_Y , train_Y )
        return loss
        '''
        ##gradient descent
        totalloss = 0
        s_gra = 0
        for i in range( iterator ):
            predict_Y = self.predict( train_X )
            diff = predict_Y - train_Y
            gra = np.zeros( np.shape(self.W) )

            for j in range( len(train_X) ):
                gra += train_X[j] * diff[j]
            gra /= len(train_X)
            s_gra += gra**2
            ada = np.sqrt(s_gra)
            self.W = self.W  - lr_rate*gra/ada
            loss = MSE(predict_Y,train_Y)
            print( "iterator  :: " , i )
            print( "loss      :: " , loss )
            totalloss += loss
        return totalloss/iterator


    def predict(self, test_X):
        #TODO
        #predict_Y = ...?
        predict_Y = []
        for i in range( len(test_X) ):
            predict_Y.append( [np.matmul( np.squeeze(test_X[i]), np.squeeze(self.W) )])
        return predict_Y
def MSE(predict_Y, real_Y):
    #TODO :mean square error
    # loss = ?
    loss = 0
    for i in range( len( predict_Y ) ):
        loss += (predict_Y[i] - real_Y[i])**2
    loss = loss**(1/2)
    loss /= len( predict_Y )
    return loss

def plotting( training_set_loss, test_set_loss ):
    assert len( training_set_loss ) == len( test_set_loss )
    length = len( training_set_loss )
    plt.figure( figsize=(12,8) )
    plt.xticks( range( 1, length+1) )
    plt.plot( range( 1 , length+1) , training_set_loss ,'b' , label = 'train loss' )
    plt.plot( range( 1 , length+1) , test_set_loss ,'r' , label = 'test loss' )
    plt.legend()
    plt.xlabel('N')
    plt.ylabel( 'MSE loss' )
    plt.show()


if __name__ == '__main__' :
    N = 6
    training =[]
    testing = []
    for i in range( 48 ):
        N = i+1
        print( "==================================N is " , N )
        train_X, train_Y = read_TrainData('train.csv', N=N)
        model = Linear_Regression()
        train_loss = model.train(train_X, train_Y)
        print( "==================================training loss is ", train_loss )
        test_X, test_Y = read_TestData('test.csv', N=N)
        #print(train_X , train_Y)
        #print( test_X , test_Y )
        predict_Y = model.predict(test_X)
        test_loss = MSE(predict_Y, test_Y)
        print("==================================testing loss is  " ,test_loss)
        training.append( train_loss )
        testing.append( test_loss )
    plotting( training , testing )

