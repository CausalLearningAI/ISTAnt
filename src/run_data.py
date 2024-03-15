from data import load_data
from model import get_embeddings
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_proc", type=int, default=5, help="Number of processes")
    parser.add_argument("--environment", type=str, default="supervised", help="Environment")
    parser.add_argument("--generate", type=bool, default=False, help="Generate the dataset")
    parser.add_argument("--reduce_fps_factor", type=int, default=15, help="Reduce fps factor")
    parser.add_argument("--downscale_factor", type=float, default=1, help="Downscale factor")
    return parser


def main(args):
    data = load_data(
        environment=args.environment,
        data_dir=args.data_dir,
        generate=args.generate,
        reduce_fps_factor=args.reduce_fps_factor,
        downscale_factor=args.downscale_factor,
    )
    print("Data generated")

    encoder_names = ["vit", "dino", "clip"]
    for encoder_name in encoder_names:
        get_embeddings(
            data,
            encoder_name,
            environment=args.environment,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            data_dir=args.data_dir,
        ) 
        print(f"Embedding ({encoder_name}) added")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
