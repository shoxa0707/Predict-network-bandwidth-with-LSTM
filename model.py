# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pan
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import joblib

class TimeSeriesModelCreator(object):
    def __init__(self, look_back=1, path='datasets/times&bands.csv'):
        self.look_back = look_back
        self.current_data = self._load_dataset(path)
        self.models = {}
        self.scalers = {}

    def _load_dataset(self, path):
        #name = r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv'
        df = pan.read_csv(path)
        df = df.sort_values('timestamp')
        return df

    # convert an array of values into a dataset matrix
    def _create_dataset(self, dataset, shift):
        dataX, dataY = [], []
        for i in range(len(dataset) - (self.look_back + shift)):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back + shift, 0])
        return numpy.array(dataX), numpy.array(dataY)

    def _create_dataset_predict(self, dataset):
        dataX = []
        for i in range(len(dataset) - (self.look_back-1)):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
        return numpy.array(dataX)

        
    def _create_model(self, node, layers):
        model = Sequential()
        #add as many layers as specified in parameter but ...
        for _ in range(layers - 1):
            model.add(LSTM(node, return_sequences=True, input_shape=(self.look_back, 1)))

        #... the last layer is always this one.
        model.add(LSTM(node, input_shape=(self.look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

        return model

    
    def train_model(self, nodes, layers, path):
        for node in nodes:
            for source in range(1, 24):
                name = str(node)+"_"+str(source)
                destination = 11
                model = self._create_model(node, layers)
                local_dataframe = self.current_data[(self.current_data.source == source) & (self.current_data.destination == destination)][['bandwidth']]

                dataset = local_dataframe.values

                # normalize the dataset
                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(dataset)
                scaler_filename = f"{path}/scalers/scaler{name}.save"
                joblib.dump(scaler, scaler_filename)                
                # split into train and test sets
                # reshape into X=t and Y=t+1
                trainX, trainY = self._create_dataset(dataset, 0)

                # reshape input to be [samples, time steps, features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
                print(f"\nTraining lstm node={node} source={source} (destination always {destination})")
                model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.3)
                model.save(f"{path}/models/lstm{name}.h5")


    def predict(self, model, scaler, dataframe):
        dataset = dataframe.values

        # normalize the dataset
        dataset = scaler.transform(dataset)

        # split into train and test sets
        test = dataset

        # reshape into X=t and Y=t+1
        testX = self._create_dataset_predict(test)

        # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # make predictions
        testPredict = model.predict(testX)

        # invert predictions
        testPredict = scaler.inverse_transform(testPredict)

        return testPredict
