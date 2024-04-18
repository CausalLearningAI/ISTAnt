import argparse
import pandas as pd
import time

from data import PPCI
from utils import get_time_string

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/istant_lq", help="Path to the data directory")
    parser.add_argument("--results_dir", type=str, default="./results/istant_lq", help="Path to the results directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_nodes", type=int, default=256, help="Number of nodes per hidden layer")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--num_proc", type=int, default=6, help="Number of processes")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose")

    return parser

def main(args):
    encoders = ["dino", "clip_large", "clip", "mae", "vit", "vit_large"]
    tokens = ["class", "mean", "all"]
    split_criterias = ["experiment", "experiment_easy", "position", "position_easy", "random", "random_easy"]
    tasks = ["all", "or"] #, "yellow", "blue", "sum"]
    
    hidden_layerss = [1,2]
    lrs = [0.05, 0.005, 0.0005]
    seeds = [0, 1, 2, 3, 4]
    colors = ["yellow", "blue"]

    n_exp = len(encoders)*len(tokens)*len(tasks)*len(split_criterias)*len(hidden_layerss)*len(lrs)*len(seeds)
    k = 0
    start_time = time.time()
    results = pd.DataFrame(columns=["encoder", "token", "split_criteria", "hidden_layers", "task", "lr", "seed", "color", 'train', 'acc', 'balanced_acc', 'bias', 'bias_d', 'oe_co', 'oe_tr', 'tr_equality_control', 'tr_equality_treatment', 'ead', 'aipw', 'ead_hat', 'ead_hat_d', 'aipw_hat', 'aipw_hat_d', 'best_epoch'])
    for encoder in encoders:
        print("Encoder: ", encoder)
        for token in tokens:
            print("Token: ", token)
            for split_criteria in split_criterias:
                print("Split Criteria: ", split_criteria)
                for task in tasks:
                    print("Task: ", task)
                    dataset = PPCI(encoder = encoder,
                                token = token,
                                task = task,
                                split_criteria = split_criteria,
                                environment = "supervised",
                                batch_size = args.batch_size, 
                                num_proc = args.num_proc,
                                data_dir = args.data_dir,
                                results_dir = args.results_dir,
                                verbose = args.verbose)
                    for hidden_layers in hidden_layerss:
                        print("Hidden Layers: ", hidden_layers)
                        for lr in lrs:
                            print("Learning Rate: ", lr)
                            for seed in seeds: 
                                print("Seed: ", seed)
                                k +=1
                                start_time_i = time.time()
                                dataset.train(batch_size = args.batch_size,
                                            num_epochs = args.num_epochs,
                                            lr = lr,
                                            add_pred_env = "supervised",
                                            seed = seed,
                                            hidden_layers = hidden_layers,
                                            hidden_nodes = args.hidden_nodes, 
                                            verbose = args.verbose,
                                            save = True)
                                end_time_i = time.time()
                                print(f"Experiment {k}/{n_exp} completed; Speed: {round(end_time_i-start_time_i, 1)}s/train, Total time elapsed {get_time_string(end_time_i - start_time)} (out of {get_time_string((end_time_i - start_time)/k*n_exp)}).")
                                
                                for train in [True, False]:
                                    if task == "all":
                                        for color in colors:
                                            result = dataset.evaluate(color=color, train=train, verbose=False)
                                            result["encoder"] = encoder
                                            result["token"] = token
                                            result["split_criteria"] = split_criteria
                                            result["hidden_layers"] = hidden_layers
                                            result["task"] = task
                                            result["lr"] = lr
                                            result["seed"] = seed
                                            result["color"] = color
                                            result["train"] = train
                                            result["best_epoch"] = dataset.model.best_epoch
                                            results.loc[len(results.index)] = result
                                    else:
                                        result = dataset.evaluate(color=task, train=train, verbose=False)
                                        result["encoder"] = encoder
                                        result["token"] = token
                                        result["split_criteria"] = split_criteria
                                        result["hidden_layers"] = hidden_layers
                                        result["task"] = task
                                        result["lr"] = lr
                                        result["seed"] = seed
                                        result["color"] = task
                                        result["train"] = train
                                        result["best_epoch"] = dataset.model.best_epoch
                                        results.loc[len(results.index)] = result
    
    results.to_csv(f"{args.results_dir}/experiments_result.csv")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    


