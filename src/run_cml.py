import argparse
from data import get_data_sl, get_data_cl
from model import train_model, compute_ead
from utils import set_seed


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data_dir", type=str, default="./data/", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="dino", help="Model name")
    parser.add_argument("--outcome", type=str, default="sum", help="Outcome")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes")
    parser.add_argument("--environment", type=str, default="train", help="Environment")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    return parser


def main(args):
    set_seed(args.seed)

    print("Loading data")
    X, y = get_data_sl(environment=args.environment, 
                       model_name=args.model_name, 
                       outcome=args.outcome)
    
    print("Training Model")
    model = train_model(X, y, 
                        test_size=args.test_size, 
                        batch_size=args.batch_size, 
                        num_epochs=args.num_epochs, 
                        lr=args.lr, 
                        verbose=args.verbose)

    print("ATE Estimation")
    X, y, t = get_data_cl(environment=args.environment, 
                          model_name=args.model_name, 
                          outcome=args.outcome)
    ate = compute_ead(y, t)
    ate_ml = compute_ead(model(X).sigmoid(), t)
    # ate_cl = compute_ate(y, t, X) TODO
    print(f"ATE (GT): {ate}")
    print(f"ATE (ML): {ate_ml}")
    # print(f"ATE (CL): {ate_cl}") TODO

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    


