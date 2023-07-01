import tune_params
import constants
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import utils

def main():
    raw_data, data = tune_params.load_data()
    class_weights = tune_params.get_class_weights(data)
    
    well_id_map, data_dict = tune_params.get_map_well_id_name(raw_data)
    data['WELL_ID'] = data['WELL_ID'].replace(data_dict)

    # features = data.keys().drop(constants.DROP_COLUMNS)
    x_train, x_test, y_train, y_test = tune_params.split_data(data=data, 
                                                              features=constants.FEATURE_COLUMNS, 
                                                              test_size=0.3, 
                                                              shuffle=True, 
                                                              random_state=constants.MY_RANDOM_STATE)
    x_train = x_train.set_index(['DATE'])
    x_test = x_test.set_index(['DATE'])
    y_train = y_train.set_index(['DATE'])
    y_test = y_test.set_index(['DATE'])

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train = np_utils.to_categorical(encoded_Y)

    x_scaler = MinMaxScaler()
    x_train_scaled = tune_params.normalize_data(x_train, x_scaler).astype('float32')
    x_test_scaled =  tune_params.normalize_data(x_test, x_scaler).astype('float32')

    # data['WELL_ID'] = data['WELL_ID'].replace(data_dict)
    y_test['WELL_ID'] = x_test['WELL_ID']
    y_test = y_test.replace({"WELL_ID": well_id_map})
    list_well_test = list(y_test[
        ((y_test['WellFailure'] == constants.WELL_FAILURE_YES)
        | (y_test['WellFailure'] == constants.WELL_FAILURE_MANUAL_OFF) )]
        .groupby('WELL_ID').value_counts().keys())
    list_well_test = [well for well, _ in list_well_test]

    generated_params = utils.create_param_grid(constants.HIDDEN_LAYER_NUM)
    tune_params.tune_model(x_train_scaled, 
                           y_train, 
                           x_test_scaled, 
                           y_test, 
                           list_well_test,
                           x_test,
                           well_id_map,
                           class_weights=class_weights, 
                           param_grid=generated_params)


if __name__ == '__main__':
    main()