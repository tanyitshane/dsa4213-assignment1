"""Single entry point. Examples:

    python run.py prep                      # download WikiText-2, AG News, GloVe (once per machine)
    python run.py all                       # every experiment + report, in order
    python run.py ngram
    python run.py train --model lstm        # Part I (random, trainable embeddings)
    python run.py w2v                       # Part II: train own word2vec (cost reported separately)
    python run.py train --model lstm --emb w2v
    python run.py train --model lstm --emb glove
    python run.py train --model lstm --emb frozen_random   # control: frozen but NOT pretrained
    python run.py downstream                # Part III
    python run.py report                    # figures + results/summary.md

Sensitivity runs write to their own folder and never overwrite the main results:
    python run.py all --epochs 40 --tag ep40        # -> results_ep40/
    python run.py all --seed 7 --tag seed7          # -> results_seed7/

Add --smoke to any command for a fast debugging run on a laptop (writes to results_smoke/).
"""
import argparse

from lm.utils import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["prep", "ngram", "train", "w2v", "downstream", "report",
                                        "all"])
    ap.add_argument("--model", choices=["rnn", "lstm", "transformer"])
    ap.add_argument("--emb", choices=["random", "frozen_random", "w2v", "glove"], default="random")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs, for debugging")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    ap.add_argument("--seed", type=int, default=None, help="override config.SEED")
    ap.add_argument("--lr", type=float, default=None, help="override config.LR")
    ap.add_argument("--tag", default="", help="write to results_<tag>/ instead of results/")
    args = ap.parse_args()

    import config
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.seed is not None:
        config.SEED = args.seed
        config.W2V["seed"] = args.seed
    if args.lr is not None:
        config.LR = args.lr
    config.TAG = args.tag
    device = get_device(args.device)
    print(f"device: {device}{'  [SMOKE TEST]' if args.smoke else ''}")

    from lm import data, downstream, embeddings, report, train

    if args.command == "prep":
        data.prepare_data()
    elif args.command == "ngram":
        train.run_ngram(args.smoke)
    elif args.command == "train":
        if not args.model:
            ap.error("train needs --model")
        train.run_neural(args.model, args.emb, args.smoke, device)
    elif args.command == "w2v":
        print(embeddings.train_word2vec(args.smoke))
    elif args.command == "downstream":
        downstream.run_downstream(args.smoke, device)
    elif args.command == "report":
        report.make_report(args.smoke)
    elif args.command == "all":
        train.run_ngram(args.smoke)
        for arch in ["rnn", "lstm", "transformer"]:
            train.run_neural(arch, "random", args.smoke, device)
        embeddings.train_word2vec(args.smoke)
        train.run_neural("lstm", "frozen_random", args.smoke, device)
        train.run_neural("lstm", "w2v", args.smoke, device)
        train.run_neural("lstm", "glove", args.smoke, device)
        downstream.run_downstream(args.smoke, device)
        report.make_report(args.smoke)


if __name__ == "__main__":
    main()
