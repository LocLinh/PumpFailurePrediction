WELL_FAILURE_NORMAL = 0
WELL_FAILURE_YES = 1
WELL_FAILURE_MANUAL_OFF = 2

MY_RANDOM_STATE = 28

DATA_PATH = './data/data.pkl'
DAILY_TRAIN_PATH = './data/daily_train.pkl'
DAILY_TEST_PATH = './data/daily_test.pkl'
OUTPUT_DIR = './output'

BATCH_SIZE = 32
EPOCHS = 30
START = 2088
OUTPUT_CLASSES = 3

DEFAULT_HYPER_PARAMS = {
    'M1': {
        'layer_num' : 3, 
        'neurons' : [16,8,4], 
        'activation_func' : ['relu','sigmoid','softmax'],
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy'
    },
    'M2': {
        'layer_num' : 4,
        'neurons' : [32,16,8,4], 
        'activation_func' : ['relu', 'relu','relu','relu'],
        'optimizer' : 'adam',
        'loss': 'categorical_crossentropy'
    }
}

BALANCED_CLASS_WEIGHTS = {
    0: 1,
    1: 1,
    2: 1,
}

HIDDEN_LAYER_NUM = 3
NEURON_SET = set([3, 8, 16, 32])
ACTIVATION_SET = {'relu', 'tanh', 'sigmoid'}
OPTIMIZER_SET = {'adam', 'sgd'}
LOSS_SET = {'categorical_crossentropy'}

DROP_COLUMNS = ['WellFailure','PUMP','FAILURE','PIP(PSI)','AMPERAGE', 'FREC(Hz)','WHP(PSI)','MSCF','BFPD']
