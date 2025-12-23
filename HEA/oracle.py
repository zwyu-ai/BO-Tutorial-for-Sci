from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularDataset
import pandas as pd
import numpy as np


class ObjectiveWrapper:

    def __init__(self, black_box_function,
                 input_length: int | None = None,
                 input_columns: list | None = None,
                 input_list=False):

        self.black_box_function = black_box_function
        assert not (input_length is None and input_columns is None), 'Must provide input length or input column names!'

        if input_columns is None:
            input_columns = []
            assert input_length is not None
            for i in range(input_length):
                input_columns.append('x'+str(i))

        self.input_columns = input_columns
        self.input_list = input_list

    def __call__(self, *args, **kwargs):

        if args != ():
            if 'predict' not in dir(self.black_box_function):
                return self.black_box_function(*args)
            else:
                to_predict = pd.DataFrame(args[0])
                if to_predict.shape[-1] == 1:
                    to_predict = to_predict.T
                to_predict.columns = self.input_columns
                return list(self.black_box_function.predict(to_predict))

        input_data = list(kwargs.values())
        if 'predict' not in dir(self.black_box_function):
            if not self.input_list:
                return self.black_box_function(*input_data)
            else:
                return self.black_box_function(input_data)
        else:
            to_predict = pd.DataFrame(input_data).T
            to_predict.columns = self.input_columns
            return list(self.black_box_function.predict(to_predict))[0]


def get_data():
    data = pd.read_excel('./data/oracle_data.xlsx')
    data = data[['Co', 'Fe', 'Mn', 'V', 'Cu', 'target']]
    return data


def train_model():
    train_data = get_data()
    data_train_X = TabularDataset(train_data)
    predictor = TabularPredictor(label='target').fit(data_train_X)
    return predictor


class Oracle:
    def __init__(self, model_dir: str):
        self.model = TabularPredictor.load(model_dir)

    def __call__(self, x: pd.DataFrame):
        y = self.model.predict(x)
        return np.asarray(y)

if __name__ == "__main__":
    train_model()
