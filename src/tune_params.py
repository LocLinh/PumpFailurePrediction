import constants
import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input
import random
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm


def tune_model(x__train, 
               y__train, 
               x__test, 
               y__test, 
               list_well_test,
               x_test,
               well_id_map,
               class_weights=constants.BALANCED_CLASS_WEIGHTS, 
               param_grid:dict=constants.DEFAULT_HYPER_PARAMS):
    no = 0
    for key in tqdm(param_grid):
        no += 1
        if no <= constants.START:
            continue
        model = make_model((x__train.shape[1],),
                           constants.OUTPUT_CLASSES,
                           layer_num=param_grid[key]['layer_num'],
                           neurons=param_grid[key]['neurons'],
                           activation_func=param_grid[key]['activation_func'])
        
        model.compile(loss=param_grid[key]['loss'], optimizer=param_grid[key]['optimizer'], metrics=['accuracy'])
        model.fit(x__train, 
                  y__train, 
                  batch_size=constants.BATCH_SIZE, 
                  epochs=constants.EPOCHS, 
                  class_weight=class_weights, 
                  verbose=0)

        result = make_evaluate(model, x__test, y__test, list_well_test, x_test, well_id_map)
        # print(result)
        with open(constants.OUTPUT_DIR + "/output.txt", "a") as output_file:
            output_file.write(f'Model parameters of key={key} :\n') 
            output_file.write(str(param_grid[key]))
            output_file.write(f'\n{result[0]}\n') 
            output_file.write(f'{result[1]}\n') 
            output_file.write(f'{result[2]}\n') 
            output_file.write(f'{result[3]}\n')
            output_file.write("==============================\n")


def load_data():
    with open(constants.DAILY_TRAIN_PATH, 'rb') as t_train_file:
        daily_train = pickle.load(t_train_file)
    with open(constants.DAILY_TEST_PATH, 'rb') as t_test_file:
        daily_test = pickle.load(t_test_file)
    with open(constants.DATA_PATH, 'rb') as t_data_file:
        data = pickle.load(t_data_file)

    return daily_train, daily_test, data


def get_class_weights(data:DataFrame):
    weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=data['FAILURE'].unique(), 
                                                 y=data['FAILURE'].values)
    class_weights = dict(zip(data['FAILURE'].unique(), weights)) 
    return class_weights


def split_data(data:DataFrame, features:list, test_size:float, shuffle:bool=None, random_state:int=None):
    x_train, x_test, y_train, y_test = train_test_split(data[features], 
                                                        data[['DATE','FAILURE']], 
                                                        test_size=test_size, 
                                                        shuffle=shuffle, 
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test


def normalize_data(data:DataFrame, scaler:MinMaxScaler):
    return scaler.fit_transform(data)


def get_map_well_id_name(daily_train:DataFrame, daily_test:DataFrame):
    temp_data = pd.concat(objs=[daily_train, daily_test], axis=0).reset_index().drop(columns='index')
    nums = np.arange(1,len(temp_data['WELL_ID'].unique())+1)
    data_dict = dict(zip(temp_data['WELL_ID'].unique(),nums))
    well_id_map = {v: k for k, v in data_dict.items()}
    return well_id_map


def map_well_name(data:DataFrame, test_set:DataFrame, map_id_name:dict):
    data['WELL_ID'] = test_set['WELL_ID']
    data = data.replace({"WELL_ID": map_id_name})
    return data

def make_prediction(model:Model, x:DataFrame, x_test:DataFrame, map_id_name:dict):
    pred = model.predict(x)
    pred = np.argmax(pred, axis=1)
    pred = pd.DataFrame(pred,columns=['FAILURE'], index=x_test.index)
    pred = map_well_name(pred, x_test, map_id_name=map_id_name)
    return pred

def make_model(input_shape, output_shape, layer_num=3, neurons=[16,8,4], activation_func=['relu','relu','sigmoid']):
    model = Sequential()
    # input
    model.add(Input(shape=input_shape))

    # hidden layer(s)
    for i in range(layer_num):
        model.add(Dense(neurons[i], activation=activation_func[i]))
        if i <= 2:
            model.add(Dropout(0.1))

    # output
    model.add(Dense(output_shape, activation='softmax'))

    return model


def make_evaluate(model, x, y, list_well_test, x_test, map_id_name):
    print('getting prediction output: ')
    prediction = make_prediction(model, x, x_test=x_test, map_id_name=map_id_name)

    # filter sample with output 'yes' or 'manual off'
    list_well_pred = list(prediction[
        ((prediction['FAILURE'] == constants.WELL_FAILURE_YES) 
        | (prediction['FAILURE'] == constants.WELL_FAILURE_MANUAL_OFF) )]
        .groupby('WELL_ID').value_counts().keys())
    list_well_pred = [well for well, _ in list_well_pred]

    # filter well
    chosen_wells = list(set(list_well_test) & set(list_well_pred))
    if len(chosen_wells) > 15:
        random.seed(constants.MY_RANDOM_STATE)
        chosen_wells = random.sample(chosen_wells, 15)

    # get sample of pred and true
    sample_pred = get_sample(prediction, chosen_wells)
    sample_true = get_sample(y, chosen_wells)


    print('evaluating...')
    # evaluate
    acc_score = accuracy_score(sample_true['FAILURE'], sample_pred['FAILURE'])
    f1__score = f1_score(sample_true['FAILURE'], sample_pred['FAILURE'])
    delta_time = sample_true['DATE'] - sample_pred['DATE']

    date_scores = []
    for idx in range(len(sample_pred)):
        timedelta = abs(delta_time[idx].days)
        if timedelta == 0:
            date_scores.append(1) # 1
        elif timedelta == 1: 
            date_scores.append(0.9 ) # 0.9
        elif timedelta == 2:
            date_scores.append(0.8) # 0.8
        elif timedelta == 3:
            date_scores.append(0.7) # 0.7
        elif timedelta == 4:
            date_scores.append(0.6) # 0.6
        elif timedelta == 5:
            date_scores.append(0.5) # 0.5
        elif timedelta == 6:
            date_scores.append(0.4) # 0.4
        elif timedelta == 7:
            date_scores.append(0.3) # 0.3
        elif timedelta < 30:
            date_scores.append(0.2) # 0.2
        else:
            date_scores.append(0.0) # 0.0

    date_score = np.mean(date_scores)

    final_score = 0.8*f1__score + 0.2*date_score
    print('Done!\n')
    return acc_score, f1__score, date_score, final_score


def get_sample(test_set, chosen_wells):
    res = test_set[test_set['WELL_ID'].isin(chosen_wells)] \
                .query("FAILURE > 0") \
                .groupby('WELL_ID') \
                .apply(lambda x:x.sample(1, random_state=constants.MY_RANDOM_STATE)) \
                .sample(n=len(chosen_wells), random_state=constants.MY_RANDOM_STATE)

    try: 
        wells = [item[0] for item in np.array(res.FAILURE.keys())]
        time = [item[1] for item in np.array(res.FAILURE.keys())]
    except:
        wells = [res.WELL_ID[i] for i in np.array(res.WELL_ID.keys())]
        time = [i for i in np.array(res.WELL_ID.keys())]
    res = res.FAILURE.to_numpy()
    res = pd.DataFrame(list(zip(wells, time, res)), columns =['WELL_ID', 'DATE', 'FAILURE'])
    return res


