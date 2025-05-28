import argparse
import torch

import numpy as np
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, y_test = load_data()
    
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
    ### WRITE YOUR CODE HERE, revoir ca, et voir ds dataset combien de validation ils donnent normalement
        #For reproducibility : np.random.seed(42)
        N = xtrain.shape[0]#nbr d'images au total
        indices=np.arange(N)
        np.random.shuffle(indices)

        val_split = int(0.8 * N)
        train_idx=indices[:val_split]
        val_idx=indices[val_split:]

        #random add to validation choosing
        x_val, y_val = xtrain[val_idx], ytrain[val_idx]
        xtrain, ytrain = xtrain[train_idx], ytrain[train_idx]

        xtest,ytest = x_val,y_val


    #########IDK IF IM ALLOWED TO DO THIS########

    ### WRITE YOUR CODE HERE to do any other data processing


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        # MLP needs flattened input (N, 28*28*3)
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1) 
    elif args.nn_type == "cnn":
        xtrain = np.transpose(xtrain, (0, 3, 1, 2))  # for CNN
        xtest = np.transpose(xtest, (0, 3, 1, 2))
        nbr_channels=xtrain.shape[1]
        model = CNN(nbr_channels, n_classes)  # DermaMNIST has 7 classes
    
    summary(model)
    

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
