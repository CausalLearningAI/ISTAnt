import argparse
from data import get_data_sl
from train import train_model, evaluate_model
from utils import set_seed
import pandas as pd
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to the results directory")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--encoder_name", type=str, default="dino", help="Model name")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes")
    parser.add_argument("--environment", type=str, default="supervised", help="Environment")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose")
    parser.add_argument("--split_criteria", type=str, default="experiment_easy", help="Splitting criteria")
    return parser


def main(args):
    encoders = ["clip", "clip_large", "dino", "mae", "vit", "vit_large"]
    tokens = ["class", "mean", "all"]
    tasks = ["all", "blue", "yellow", "or", "sum"]
    lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
    seeds = [0, 1, 2, 3, 4]

    results = pd.DataFrame(columns=["encoder", "token", "task", "lr", "seed", "tr_precision", "tr_recall", "tr_accuracy", "val_precision", "val_recall", "val_accuracy"])
    for encoder in encoders:
        print("Encoder: ", encoder)
        for token in tokens:
            print("Token: ", token)
            if token=="all":
                if not os.path.exists(f"./data/embeddings/mean/{encoder}/{args.environment}"):
                    continue
                if not os.path.exists(f"./data/embeddings/class/{encoder}/{args.environment}"):
                    continue
            else: 
                if not os.path.exists(f"./data/embeddings/{token}/{encoder}/{args.environment}"):
                    continue
            for task in tasks:
                print("Task: ", task)
                X, y, split = get_data_sl(environment=args.environment, 
                            encoder_name=encoder, 
                            data_dir=args.data_dir,
                            task=task,
                            split_criteria=args.split_criteria,
                            token=token)
                X_train, y_train = X[split], y[split]
                X_val, y_val = X[~split], y[~split]
                y_train.task, y_val.task = y.task, y.task
                for lr in lrs:
                    print("Lr: ", lr)
                    for seed in seeds: 
                        print("Seed: ", seed)
                        set_seed(seed)
                        model = train_model(X, y,  
                                            batch_size=args.batch_size, 
                                            num_epochs=args.num_epochs, 
                                            lr=lr, 
                                            split=split,
                                            verbose=args.verbose)
                        accs_tr, precisions_tr, recalls_tr = evaluate_model(model, X_train, y_train, device=model.device)
                        accs_val, precisions_val, recalls_val = evaluate_model(model, X_val, y_val, device=model.device)
                        result = {"encoder": encoder, 
                                "token": token, 
                                "task": task, 
                                "lr": lr,
                                "seed": seed,
                                "tr_precision": round(sum(precisions_tr)/len(precisions_tr), 4), 
                                "tr_recall": round(sum(recalls_tr)/len(recalls_tr), 4), 
                                "tr_accuracy": round(sum(accs_tr)/len(accs_tr), 4),
                                "val_precision": round(sum(precisions_val)/len(precisions_val), 4), 
                                "val_recall": round(sum(recalls_val)/len(recalls_val), 4), 
                                "val_accuracy": round(sum(accs_val)/len(accs_val), 4)}
                        results.loc[len(results.index)] = result
                        
    results.to_csv("results/head_training_results.csv")
                        

    

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)