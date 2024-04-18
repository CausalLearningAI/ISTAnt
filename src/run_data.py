import argparse
from data import PPCI

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_proc", type=int, default=5, help="Number of processes")
    parser.add_argument("--environment", type=str, default="supervised", help="Environment")
    parser.add_argument("--generate", type=bool, default=False, help="Generate the dataset")
    parser.add_argument("--reduce_fps_factor", type=int, default=15, help="Reduce fps factor")
    parser.add_argument("--downscale_factor", type=float, default=1, help="Downscale factor")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    return parser

def main(args):
    encoders = ["vit", "dino", "clip", "vit_large", "clip_large", "mae"]
    for encoder in encoders:
        PPCI(encoder = encoder,
             token = "all",
             task = "all",
             split_criteria = "experiment",
             environment = "all",
             batch_size = args.batch_size, 
             num_proc = args.num_proc,
             data_dir = args.data_dir,
             verbose = args.verbose)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
