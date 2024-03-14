import argparse
from data import get_data_sl, get_data_cl
from train import train_model
from causal import compute_ead
from utils import set_seed
from visualize import visualize_examples

import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data_dir", type=str, default="./data/", help="Path to the data directory")
    parser.add_argument("--path_results_dir", type=str, default="./results/", help="Path to the results directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="vit", help="Model name")
    parser.add_argument("--outcome", type=str, default="all", help="Outcome")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes")
    parser.add_argument("--environment", type=str, default="supervised", help="Environment")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    return parser


def main(args):
    set_seed(args.seed)

    print("Loading data")
    X, y = get_data_sl(environment=args.environment, 
                       model_name=args.model_name, 
                       data_dir=args.path_data_dir,
                       outcome=args.outcome)
    
    print("Training Model")
    model = train_model(X, y, 
                        test_size=args.test_size, 
                        batch_size=args.batch_size, 
                        num_epochs=args.num_epochs, 
                        lr=args.lr, 
                        verbose=args.verbose)
    y_probs = model.probs(X)
    y_pred = model.pred(X)

    # get 36 integeres at random in [0,len(y)] with torch
    idxs = torch.randint(0, len(y), (36,))
    visualize_examples(idxs, 
                       outcome=args.outcome, 
                       model_name=args.model_name, 
                       model=model, 
                       save=True, 
                       path_results_dir="./results/")

    print("ATE Estimation")
    X, y, t = get_data_cl(environment=args.environment, 
                          data_dir=args.path_data_dir,
                          outcome=args.outcome)
    print(f"ATE (GT)")
    ate_B, ate_inf = compute_ead(y, t, verbose=args.verbose)
    print(f"ATE (ML)")
    ate_ml_B, ate_ml_inf = compute_ead(y_probs, t, verbose=args.verbose)
    print(f"ATE (ML disc.)")
    ate_ml_d_B, ate_ml_d_inf = compute_ead(y_pred, t, verbose=args.verbose)
    #print(f"ATE (CL)")
    #ate_cl_B, ate_cl_inf = compute_ate(y, t, X) TODO

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    


