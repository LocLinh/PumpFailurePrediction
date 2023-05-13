import constants
import itertools

def get_hyper_params_combination(layer_num):
    res = []
    for i in range(layer_num):
        layer_idx = i + 1
        neuron_sets = [constants.NEURON_SET] * layer_idx
        activation_sets = [constants.ACTIVATION_SET] * layer_idx
        
        # Get all combinations of the i sets
        combinations = list(itertools.product(*neuron_sets, *activation_sets, constants.OPTIMIZER_SET, constants.LOSS_SET))
        for idx, combo in enumerate(combinations):
            res.append({
                'layer_num' : layer_idx,
                'neurons' : list(combo[:layer_idx]),
                'activation_func' : list(combo[layer_idx: layer_idx*2]),
                'optimizer': combo[-2],
                'loss': combo[-1]
            })
    return res


def create_param_grid(hidden_layer_num=4):
    param_grid = {}
    for layer_num in range(1, hidden_layer_num+1, 1): # 1 -> n hidden layer
        combo_params = get_hyper_params_combination(layer_num)
        prefix_name = ''
        if layer_num == 1:
            prefix_name = 'A'
        elif layer_num == 2:
            prefix_name = 'B'
        elif layer_num == 3:
            prefix_name = 'C'
        elif layer_num == 4:
            prefix_name = 'D'
        else:
            prefix_name = '_'
        for idx, combo in enumerate(combo_params):
            key_name = f'{prefix_name}_{idx}'
            param_grid[key_name] = combo
    return param_grid