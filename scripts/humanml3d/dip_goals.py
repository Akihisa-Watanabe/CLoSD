"""DiP, CLoSD's planner, on the protocol of ARDY's Table 5: autoregressive text-to-motion with joint goals on HumanML3D,
for the benchmark's eval/humanml3d/eval_goals.py.

    python scripts/humanml3d/dip_goals.py in --model_path <DiP_multi-target_10steps_context20_predict40>/model000300000.pt \
        --prior2policy ../../../.. --out dip_goals_in.pkl

Every test item is 9 s at 20 fps (data/humanml3d_goals.pt, built by eval_goals.py): its first second of recorded motion
is the history, and DiP plans the four windows of 2 s behind it, each from the last 20 frames before it, under the
item's caption and a goal, as its stand-alone autoregressive sampling plans (closd/diffusion_planner/utils/
sampler_util.py). DiP is the goal-conditioned one CLoSD releases (DiP_multi-target), with the guidance CLoSD gives it
(closd/data/cfg/env/closd_base.yaml, cfg_param 7.5). A goal is the position of the goal's joint at the goal's frame,
read from the item's recorded motion (HumanML3D's features, in the world they place the motion in) and expressed in
the frame of the window's first frame, as get_target_location (data_loaders/humanml/scripts/motion_process.py)
expresses it during training: the position alone, as ARDY gives DiP its goals (docs/evaluation.md). DiP was trained
with a goal at the last frame of every window that always came with the body's heading there (`DIMP_FULL`), which
--heading gives too. The goal of a
window is
    in   the item's goal at the window's last frame; where the motion has ended before it, its last frame's goal;
    out  the item's one goal at the motion's last frame, in every window.
The goal joints are eval_goals.py's bodies of the SMPL humanoid by their HumanML3D joints: the pelvis, the wrists and
the feet (HumanML3D's foot joints, the toes). Writes HumanML3D's features of the 9 s, history included, normalized by
its Mean.npy and Std.npy, with the goals given and every item's length, for eval_goals.py.
"""
import argparse
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qrot
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (recover_from_ric, recover_root_rot_heading_ang,
                                                                              recover_root_rot_pos)
from closd.diffusion_planner.data_loaders.humanml_utils import HML_JOINT_NAMES
from closd.diffusion_planner.utils import dist_util
from closd.diffusion_planner.utils.fixseed import fixseed
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model
from closd.diffusion_planner.utils.parser_util import generate_args
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel

HISTORY, WINDOW, WINDOWS = 20, 40, 4
FRAMES = HISTORY + WINDOWS * WINDOW
DIP_JOINT = {"Pelvis": "pelvis", "L_Wrist": "left_wrist", "R_Wrist": "right_wrist", "L_Toe": "left_foot", "R_Toe": "right_foot"}


class NoActions:
    class dataset:
        pass


def body_names(mjcf):
    return [body.get("name") for body in ET.parse(mjcf).getroot().find("worldbody").iter("body")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("goals", choices=("in", "out"))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--guidance", type=float, default=7.5, help="CLoSD's cfg_param for its goal-conditioned DiP")
    parser.add_argument("--prior2policy", type=Path, required=True, help="the repository, for its test items and goals")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--heading", action="store_true", help="also give the body's heading, as DiP was trained")
    parser.add_argument("--out", type=Path, required=True)
    ours = parser.parse_args()
    sys.argv = [sys.argv[0], "--model_path", ours.model_path, "--guidance_param", str(ours.guidance)]
    args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args, NoActions)
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    model = ClassifierFreeSampleModel(model).to(device).eval()
    extended = model.all_goal_joint_names + ["traj", "heading"]

    root = ours.prior2policy
    test = torch.load(root / "data/humanml3d_test.pt", weights_only=False)
    goals = torch.load(root / "data/humanml3d_goals.pt", weights_only=False)
    names = body_names(root / "assets/smpl_humanoid.xml")
    joint_of = {names.index(body): HML_JOINT_NAMES.index(DIP_JOINT[body]) for body in DIP_JOINT}
    mean, std = (torch.from_numpy(np.load(root / "data/HumanML3D" / f)).float().to(device) for f in ("Mean.npy", "Std.npy"))
    count = len(test["key"])
    if ours.goals == "in":  # a window's goal; where the motion has ended, its last frame's
        frame = torch.where(goals["in_frame"] >= 0, goals["in_frame"], goals["out_frame"][:, None])
        body = torch.where(goals["in_frame"] >= 0, goals["in_body"], goals["out_body"][:, None])
    else:
        frame, body = goals["out_frame"][:, None].expand(-1, WINDOWS), goals["out_body"][:, None].expand(-1, WINDOWS)

    episodes = []
    for start in tqdm(range(0, count, ours.batch_size)):
        items = list(range(start, min(start + ours.batch_size, count)))
        bs = len(items)
        recorded = torch.stack([test["motion"][i][:FRAMES] if len(test["motion"][i]) >= FRAMES else
                                torch.cat([test["motion"][i], test["motion"][i][-1:].expand(FRAMES - len(test["motion"][i]), -1)])
                                for i in items]).float().to(device)  # [bs, 180, 263], the last frame held
        world = recover_from_ric(recorded, 22)  # [bs, 180, 22, 3], the joints of the recorded motion
        text = [test["captions"][i][goals["caption"][i]]["text"] for i in items]
        text_embed = model.encode_text(text)
        motion = recorded[:, :HISTORY]
        rows = torch.arange(bs, device=device)
        for k in range(WINDOWS):
            # the window's first frame: where the motion so far, with its last frame's velocities, puts the root
            quat, pos = recover_root_rot_pos(torch.cat([motion, motion[:, -1:]], dim=1))
            quat, pos = quat[:, -1], pos[:, -1]
            f = frame[items, k].to(device)
            local = world[rows, f] - torch.stack([pos[:, 0], torch.zeros_like(pos[:, 0]), pos[:, 2]], -1)[:, None]
            local = qrot(quat[:, None].expand(-1, 22, -1).contiguous(), local)  # [bs, 22, 3] in the window's frame
            heading = recover_root_rot_heading_ang(local[..., None])[:, 0, 0]
            target = torch.zeros(bs, len(extended), 3, device=device)
            joint_names = []
            for b, j in enumerate(body[items, k].tolist()):
                name = HML_JOINT_NAMES[joint_of[j]]
                target[b, extended.index(name)] = local[b, joint_of[j]]
                joint_names.append([name])
            if ours.heading:  # an entry not given is zero, as in training
                target[:, extended.index("heading"), 0] = heading
            prefix = ((motion[:, -HISTORY:] - mean) / std).permute(0, 2, 1).unsqueeze(2)  # [bs, 263, 1, 20]
            y = dict(text=text, text_embed=text_embed, prefix=prefix, lengths=torch.full((bs,), WINDOW),
                     mask=torch.ones(bs, 1, 1, WINDOW, dtype=torch.bool, device=device),
                     scale=torch.full((bs,), ours.guidance, device=device),
                     target_cond=target, target_joint_names=joint_names, is_heading=torch.full((bs,), ours.heading, device=device))
            with torch.no_grad():
                window = diffusion.p_sample_loop(model, (bs, model.njoints, model.nfeats, WINDOW), clip_denoised=False,
                                                 model_kwargs=dict(y=y), skip_timesteps=0, init_image=None, progress=False,
                                                 dump_steps=None, noise=None, const_noise=False)
            window = window[..., -WINDOW:]  # the sample is the context and the window
            motion = torch.cat([motion, window.squeeze(2).permute(0, 2, 1) * std + mean], dim=1)
        episodes.append(((motion - mean) / std).cpu())
    ours.out.parent.mkdir(parents=True, exist_ok=True)
    with open(ours.out, "wb") as file:
        pickle.dump(dict(motion=torch.cat(episodes), goals=ours.goals, length=goals["length"].tolist()), file, protocol=4)
    print(f"{count} episodes of {FRAMES} frames, {ours.goals}-horizon goals: {ours.out}")


if __name__ == "__main__":
    main()
