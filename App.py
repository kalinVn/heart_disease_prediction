import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class App:

    def __init__(self, csv_path):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)

        self.model = LogisticRegression()
        self.x_test_prediction = None
        self.x_train_prediction = None

    def standardize_data(self):
        self.x = self.dataset.drop(['target'], axis=1)
        self.y = self.dataset['target']

    def fit(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=2, stratify=self.y)

        self.model.fit(self.x_train, self.y_train)

    def set_x_predictions(self):
        self.x_test_prediction = self.model.predict(self.x_test)
        self.x_train_prediction = self.model.predict(self.x_train)

    def accuracy_score(self):
        training_data_accuracy = accuracy_score(self.x_train_prediction, self.y_train)
        print('Accuracy score on training data: ', training_data_accuracy)

        x_test_prediction = self.model.predict(self.x_test)
        # print(x_test_prediction)

        test_data_accuracy = accuracy_score(self.x_test_prediction, self.y_test)
        print('Accuracy score on test data: ', test_data_accuracy)

    def predict(self, input_data):

        # changing the input data to numpy array
        input_data_numpy_arr = np.asarray(input_data)
        # print(input_data_numpy_arr)

        # reshape the np array
        input_data_reshaped = input_data_numpy_arr.reshape(1, -1)

        prediction = self.model.predict(input_data_reshaped)

        if prediction[0] == 1:
            print('The person have a heart disease.')
        else:
            print('The person does not have a heart disease.')
