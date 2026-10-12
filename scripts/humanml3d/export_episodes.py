"""CLoSD's recorded episodes as the benchmark's episodes: the simulated bodies of every episode, with its test item and caption.

    python scripts/humanml3d/export_episodes.py <recorded>/CLoSD.pkl --test <prior2policy>/data/humanml3d_test.pt --out closd.npz

Reads the file that closd/run.py records with env.save_motion.save_hml_episodes=True, which on this branch also holds
the bodies' positions in the simulator (z up, the humanoid's body order, 30 steps per second). Writes an .npz of
body_pos (episodes, steps, 24, 3), fps, key (the test item of an episode) and caption (which of the item's captions
prompted it), the format the benchmark's scorer reads.
"""
import argparse
import pickle

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recorded")
    parser.add_argument("--test", required=True, help="the benchmark's test items")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.recorded, "rb") as file:
        recorded = pickle.load(file)
    test = torch.load(args.test)
    captions = {key: [caption["text"] for caption in item] for key, item in zip(test["key"], test["captions"])}
    caption = np.array([captions[key].index(text) for key, text in zip(recorded["db_key"], recorded["caption"])])
    np.savez(args.out, body_pos=recorded["body_pos"].numpy(), fps=30., key=np.array(recorded["db_key"]), caption=caption)
    saved, terminated = len(caption), recorded["terminated_episodes"]
    print(f"{saved} episodes of {recorded['body_pos'].shape[1]} steps for {len(set(recorded['db_key']))} of {len(captions)} test items; "
          f"{terminated} more ended in a termination and were not recorded ({terminated / (saved + terminated):.1%})")


if __name__ == "__main__":
    main()
