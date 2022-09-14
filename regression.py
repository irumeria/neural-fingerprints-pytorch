# Example regression script using neural fingerprints.

import torch
from torch import nn
import torch.nn.functional as F
from neural_fingerprints.io_utils import load_data
from neural_fingerprints.convnet import NeuralConvNetwork
from neural_fingerprints.deepnet import DeepNetwork

task_params = {'target_name': 'measured log solubility in mols per litre',
               'data_file': 'delaney.csv'}
N_train = 800
N_val = 20
N_test = 20

model_params = dict(fp_length=50,    # Usually neural fps need far fewer dimensions than morgan.
                    # The depth of the network equals the fingerprint radius.
                    fp_depth=4,
                    conv_width=20,   # Only the neural fps need this parameter.
                    # Size of hidden layer of network on top of fps.
                    h1_size=100)
train_params = dict(num_iters=100,
                    batch_size=100)

# Define the architecture of the network that sits on top of the fingerprints.
net_params = dict(
    # One hidden layer.
    layer_sizes=[model_params['fp_length'], model_params['h1_size']],
    normalize=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def normalize_tensor(A):
    mean, std = torch.mean(A), torch.std(A)
    A_normed = (A - mean) / std

    def restore_function(X):
        return X * std + mean

    return A_normed, restore_function


def get_ith_minibatch_ixs(i, num_datapoints, batch_size):
    num_minibatches = int(num_datapoints / batch_size +
                          ((num_datapoints % batch_size) > 0))
    i = i % num_minibatches
    start = i * batch_size
    stop = start + batch_size
    return slice(start, stop)


def main():
    print("Loading data...")
    traindata, valdata, _testdata = load_data(
        task_params['data_file'], (N_train, N_val, N_test),
        input_name='smiles', target_name=task_params['target_name'])

    def run_conv_experiment():
        train_inputs, train_targets = traindata
        val_inputs,   val_targets = valdata
        # test_inputs,  test_targets = testdata

        conv_layer_sizes = [model_params['conv_width']] * \
            model_params['fp_depth']
        conv_params = {'num_hidden_features': conv_layer_sizes,
                       'fp_length': model_params['fp_length'], 'normalize': 1}

        model_fp = NeuralConvNetwork(**conv_params)
        model_deep = DeepNetwork(**net_params)
        print(model_fp)

        device = torch.device("cuda:0" if (
            torch.cuda.is_available()) else "cpu")
        print(f"Using {device} device")

        model_fp = model_fp.apply(weights_init)
        model_deep = model_deep.apply(weights_init)

        model_total = nn.Sequential(model_fp, model_deep).to(device)
        optimizer = torch.optim.Adam(model_total.parameters(), lr=1e-2)

        train_targets, undo_norm = normalize_tensor(
            torch.from_numpy(train_targets).to(device).float())
        val_targets = torch.from_numpy(val_targets).to(device)

        for iter in range(train_params['num_iters']):
            
            batch_index = 0
            num_datapoints = len(train_inputs)
            loss_average = 0.

            while num_datapoints > batch_index*train_params['batch_size']:
                cur_idxs = get_ith_minibatch_ixs(
                    batch_index, num_datapoints, train_params['batch_size'])

                inputs = train_inputs[cur_idxs]
                targets = train_targets[cur_idxs]

                logits = model_total(inputs)
                loss = F.mse_loss(logits, targets)

                model_total.zero_grad()
                loss.backward()
                optimizer.step()

                batch_index += 1
                loss_average += train_params['batch_size'] / \
                    num_datapoints*loss.item()
            
            if iter%10 == 0:
                pred = undo_norm(model_total(val_inputs))
                print("iter", str(iter), "finished")
                print("val loss:", F.mse_loss(pred,val_targets).item())

    print("Task params", task_params)
    print("Starting neural fingerprint experiment...")
    test_loss_neural = run_conv_experiment()
    print("Neural test RMSE:", test_loss_neural)
    
if __name__ == '__main__':
    main()
