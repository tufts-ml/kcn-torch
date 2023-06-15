import numpy as np
import torch
from argument import parse_opt 
from experiment import run_kcn


# repeat the experiment in the paper
def random_runs(args):
    test_errors = []
    for args.random_seed in range(10):
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    
        err = run_kcn(args)
        test_errors.append(err)
    
    test_errors = np.array(test_errors)
    return test_errors
 


if __name__ == "__main__":

    args = parse_opt()
    print(args)

    # set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # run experiment on one train-test split
    err = run_kcn(args)
    print('Model: {}, test error: {}\n'.format(args.model, err))
    
    
    ## run all experiments on one dataset
    #model_error = dict()
    #for args.model in ["kcn", "kcn_gat", "kcn_sage"]:
    #    test_errors = random_runs(args)
    #    model_error[args.model] = (np.mean(test_errors), np.std(test_errors))
    #    print(model_error)


