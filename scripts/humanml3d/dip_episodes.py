"""CLoSD's planner, DiP, alone, without the simulator: the episodes of HumanML3D's test items as CLoSD's text-to-motion
task plans them, for the benchmark's scorer.

    python scripts/humanml3d/dip_episodes.py --model_path <DiP_no-target_10steps_context20_predict40>/model000200000.pt \
        --test <prior2policy>/data/humanml3d_test.pt --mean_std <prior2policy>/data/HumanML3D --out dip.pkl

The checkpoint and the guidance scale are those of CLoSD's text-to-motion task (closd/data/cfg/env/closd_t2m.yaml:
model000200000.pt, cfg_param 5). As CLoSD starts an episode, an item's first two seconds of recorded motion are its
prefix, of which DiP is given the last second (its context of 20 frames) under the item's caption; DiP then plans 40
frames at a time, each time from the last 20 it planned, as its stand-alone autoregressive sampling does
(closd/diffusion_planner/utils/sampler_util.py), until --frames. The prefix is not part of the episode, as it is not in
DiP's stand-alone evaluation. The caption of an item is drawn with --seed as the benchmark's prompts are (the UniPhys
fork's scripts/humanml3d/prompts.py). Writes the episodes as CLoSD saves its own, HumanML3D's features normalized by
its Mean.npy and Std.npy, with each item's key, caption, tokens and the length of its recorded motion: the
`external_results_file` the scorer reads.
"""
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model
from closd.diffusion_planner.utils.parser_util import generate_args
from closd.diffusion_planner.utils.sampler_util import AutoRegressiveSampler, ClassifierFreeSampleModel

PREFIX = 40  # frames of recorded motion an episode starts from, CLoSD's hml_data_buf_size


class NoActions:
    class dataset:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--guidance", type=float, default=5., help="CLoSD's cfg_param for text-to-motion")
    parser.add_argument("--test", type=Path, required=True, help="the benchmark's test items")
    parser.add_argument("--mean_std", type=Path, required=True, help="HumanML3D's Mean.npy and Std.npy, as CLoSD has them")
    parser.add_argument("--frames", type=int, default=200, help="planned frames at 20 fps, 10 s as CLoSD's episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    ours = parser.parse_args()
    sys.argv = [sys.argv[0], "--model_path", ours.model_path, "--guidance_param", str(ours.guidance), "--autoregressive"]
    args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    model, diffusion = create_model_and_diffusion(args, NoActions)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    model = ClassifierFreeSampleModel(model).to(dist_util.dev()).eval()
    sample = AutoRegressiveSampler(args, diffusion.p_sample_loop, ours.frames).sample

    test = torch.load(ours.test, weights_only=False)
    rng = random.Random(ours.seed)
    chosen = [rng.randrange(len(captions)) for captions in test["captions"]]
    mean, std = (torch.from_numpy(np.load(ours.mean_std / name)).float() for name in ("Mean.npy", "Std.npy"))
    motions, captions, tokens = [], [], []
    for start in tqdm(range(0, len(test["key"]), ours.batch_size)):
        items = range(start, min(start + ours.batch_size, len(test["key"])))
        prefix = torch.stack([(test["motion"][i][:PREFIX].float() - mean) / std for i in items])  # [bs, 40, 263]
        prefix = prefix[:, -args.context_len:].permute(0, 2, 1).unsqueeze(2)  # [bs, 263, 1, 20]
        text = [test["captions"][i][chosen[i]]["text"] for i in items]
        bs = len(text)
        y = dict(text=text, prefix=prefix.to(dist_util.dev()), lengths=torch.full((bs,), args.pred_len),
                 mask=torch.ones(bs, 1, 1, args.pred_len, dtype=torch.bool, device=dist_util.dev()),
                 scale=torch.full((bs,), ours.guidance, device=dist_util.dev()))
        with torch.no_grad():
            motion = sample(model, (bs, model.njoints, model.nfeats, ours.frames), clip_denoised=False,
                            model_kwargs=dict(y=y), skip_timesteps=0, init_image=None, progress=False,
                            dump_steps=None, noise=None, const_noise=False)
        motions.append(motion.squeeze(2).permute(0, 2, 1).cpu())  # [bs, frames, 263]
        captions += text
        tokens += ["_".join(["sos/OTHER"] + test["captions"][i][chosen[i]]["tokens"] + ["eos/OTHER"]) for i in items]
    ours.out.parent.mkdir(parents=True, exist_ok=True)
    with open(ours.out, "wb") as file:
        pickle.dump(dict(motion=torch.cat(motions), caption=captions, tokens=tokens, db_key=list(test["key"]),
                         length=[len(m) for m in test["motion"]]), file, protocol=4)
    print(f"{len(captions)} episodes of {ours.frames} frames: {ours.out}")


if __name__ == "__main__":
    main()
