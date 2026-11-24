"""CLoSD's tracker carrying out the plans of mvp-prior2policy's prior in place of DiP's: the prior Stage 2 starts from
(conf/prior/smpl/causal_forcing.yaml) planning, and the tracker CLoSD trained to follow plans, on the protocols
closd_goals.py plays CLoSD on: env=closd_prior for ARDY's Table 5, env=closd_prior_caption for Experiment 2.

    python closd/run.py learning=im_big robot=smpl_humanoid env=closd_prior exp_name=CLoSD_multitask_finetune epoch=-1 \
        test=True headless=True no_virtual_display=True env.num_envs=291 env.prior2policy=<mvp-prior2policy> env.out=goals_in.npz

The prior plans as Stage 2's policy decides (mvp-prior2policy's models/planner.py, Planner, as
conf/policy/smpl/causal_forcing_generative/env.yaml sets it), 2 frames at a time at 20 fps from fresh noise behind the
motion so far, its last 160 frames, under the item's caption and, on Table 5, the item's goals as the policy is given
them (data/references/smpl/causal_forcing_test.pt and _caption.pt, the policy's test references). At a decision, every
plan_frames frames, it plans plan_frames frames, each 2 behind the ones it planned before them: plan_frames 2 decides as
the policy decides, every 3 control steps; 40 plans 2 s every 2 s, as CLoSD plans. The motion so far is the
references' frames up to the start, the item's recorded second in front of it, then the simulated bodies, a frame at
the end of every frame: at a control step's end its state, between two control steps halfway between their states.
The tracker follows the plan's bodies, interpolated linearly at the control steps from the bodies at the decision, as
the policy's targets are (targets: linear). An episode starts, runs and is recorded as closd_goals.py's; DiP is loaded
with the task and not used.
"""
import sys
from pathlib import Path

import torch

from closd.env.tasks import closd_goals
from closd.utils import torch_utils
from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonState

FRAMES, HISTORY = 2, 160  # the frames a plan of the prior carries out at its 20 fps, as the policy's decision; the frames it reads

def halfway(a, b):
    """Unit quaternions a[..., 4], b[..., 4] -> the rotation halfway between them, slerp at 0.5."""
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    return torch.nn.functional.normalize(a + b, dim=-1)


class CLoSDPrior(closd_goals.CLoSDGoals):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        root = Path(cfg["env"]["prior2policy"])
        sys.path.insert(0, str(root))
        from models.ardy_model import load_model
        from models.planner import Planner
        from utils.config import load_config
        from utils.mujoco import SmplDofConverter
        prior = load_config(root / "conf/prior/smpl/causal_forcing.yaml")
        prior.update({name: str(root / prior[name]) for name in ("checkpoint", "tokenizer", "mjcf")})
        ardy = load_model(prior, torch.device(self.device))
        self.planner = Planner(ardy, SmplDofConverter(ardy.motion_rep.skeleton, prior["mjcf"]), HISTORY, FRAMES)
        references = torch.load(str(root / f"data/references/smpl/causal_forcing_test{'_caption' if self.caption_only else ''}.pt"),
                                weights_only=True, mmap=True)
        if references["motions"] != self.key.tolist() or [references["captions"][t] for t in references["text"]] != self.prompts:
            raise ValueError("the references are not the test items under the captions the task draws")
        self.first = references["first"]  # the start, the references' frame of the recorded one in front
        self.recorded = references["qpos"][:, :self.first + 1].to(self.device)
        self.text_feat = references["text_feat"][references["text"]].to(self.device)
        self.motion_mask, self.observed = references["motion_mask"], references["observed"]  # on the CPU, a round's moved over
        self.plan_frames = cfg["env"]["plan_frames"]  # the frames planned at a decision, at 20 fps
        self.plan_steps = self.plan_frames * 3 // 2  # the control steps they last at 30 Hz
        if self.plan_frames % FRAMES or (self.max_episode_length - 1) % self.plan_steps:
            raise ValueError(f"plan_frames {self.plan_frames} is not whole plans of the prior and decisions of an episode")
        self.motion = torch.zeros(self.num_envs, self.first + 1 + (self.max_episode_length - 1) * 2 // 3, self.recorded.shape[-1], device=self.device)
        self.root_pos = torch.zeros(self.num_envs, self.max_episode_length, 3, device=self.device)  # the bodies at every control step
        self.local_rot = torch.zeros(self.num_envs, self.max_episode_length, len(self.skeleton_trees[0].node_names), 4, device=self.device)
        self.planned = None  # the (round, step) of the latest decision
        self.parents = self.skeleton_trees[0].parent_indices.to(self.device)

    def body_pos(self, qpos):
        """The robot's qpos[..., 76] (mvp-prior2policy's utils/mujoco.py, SmplDofConverter) -> its bodies' positions[..., 24, 3]."""
        rot = torch.cat([qpos[..., None, 3:7], torch_utils.exp_map_to_quat(qpos[..., 7:].unflatten(-1, (-1, 3)))], -2)
        state = SkeletonState.from_rotation_and_root_translation(self.skeleton_trees[0], rot.flatten(0, -3).cpu(),
                                                                 qpos[..., :3].flatten(0, -2).cpu(), is_local=True)
        return state.global_translation.view(*qpos.shape[:-1], -1, 3).to(qpos.device)

    def qpos(self, root_pos, local_rot):
        return torch.cat([root_pos, local_rot[:, 0], torch_utils.quat_to_exp_map(local_rot[:, 1:]).flatten(1)], -1)

    def get_pred_pose(self):
        step = int(self.progress_buf[0])  # the worlds step together
        rot = self._rigid_body_rot
        self.root_pos[:, step] = self._rigid_body_pos[:, 0]
        self.local_rot[:, step] = torch.cat([rot[:, :1], torch_utils.quat_mul(torch_utils.quat_conjugate(rot[:, self.parents[1:]]), rot[:, 1:])], 1)
        if step % self.plan_steps == 0 and step < self.max_episode_length - 1 and self.planned != (self.round, step):
            self.planned = (self.round, step)
            frame = self.first + step * 2 // 3
            if step == 0:  # a round's items
                self.motion[:, :self.first + 1] = self.recorded[self.item]
                items = self.item.cpu()
                self.constraints = None if self.caption_only else (self.motion_mask[items].to(self.device), self.observed[items].to(self.device))
            else:  # the frames since the decision before, at a control step's end its state, between two halfway
                for f in range(frame - self.plan_frames + 1, frame + 1):
                    a, b = (f - self.first) * 3 // 2, (f - self.first) * 3 // 2 + (f - self.first) % 2
                    self.motion[:, f] = self.qpos((self.root_pos[:, a] + self.root_pos[:, b]) / 2, halfway(self.local_rot[:, a], self.local_rot[:, b]))
            for f in range(frame, frame + self.plan_frames, FRAMES):  # the plan, behind its own frames
                at = torch.full((self.num_envs,), f, device=self.device)
                self.motion[:, f + 1:f + 1 + FRAMES] = self.planner(self.motion, at, self.text_feat[self.item], self.planner.noise(self.num_envs), self.constraints)
            now = self._rigid_body_pos.clone()
            bodies = torch.cat([now[:, None], self.body_pos(self.motion[:, frame + 1:frame + 1 + self.plan_frames])], 1)
            at = torch.arange(1, self.plan_steps + 1, device=self.device) * 2 / 3  # the control steps' ends, in frames from the decision
            low = at.floor().long()
            high, weight = (low + 1).clamp(max=self.plan_frames), (at - low)[:, None, None]
            self.targets = self.rep.sim_pose_to_ref_pose(bodies[:, low] + (bodies[:, high] - bodies[:, low]) * weight)
            self.decision_bodies = self.rep.sim_pose_to_ref_pose(now)
        pose = {"j3d": self.targets[:, step % self.plan_steps], "dt": self.dt}
        if step % self.plan_steps == 0:  # the velocity of the first target from the bodies, as CLoSD's plans start
            pose["j3d_prev"] = self.decision_bodies
        return pose
