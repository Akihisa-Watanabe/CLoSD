"""CLoSD on the protocol of ARDY's Table 5: autoregressive text-to-motion with joint goals on HumanML3D, in the
simulator, for the benchmark's eval/humanml3d/eval_goals.py and eval_t2m.py (mvp-prior2policy).

    python closd/run.py learning=im_big robot=smpl_humanoid env=closd_goals exp_name=CLoSD_multitask_finetune epoch=-1 \
        test=True headless=True no_virtual_display=True env.num_envs=1162 env.prior2policy=<mvp-prior2policy> env.out=goals_in.npz

Every test item (data/humanml3d_goals.pt, built by eval_goals.py) is one episode, and the worlds perform them in rounds:
world w of round r performs item r * num_envs + w. A world starts at the item's recorded state at the end of its first
second, the humanoid's pose at the second's last frame (20 fps) with its velocities from the frames around it, as PHC's
motion library gives a reference state. DiP plans the four windows of 2 s behind it under the item's caption, as CLoSD
plans (closd.py): the first window from the recorded second, HumanML3D's features of the item, and the others from the
last second the simulator ran; the tracker follows each plan for 60 control steps. A window's goal is given as
scripts/humanml3d/dip_goals.py gives DiP its goals: the position of the goal's joint at the window's last frame, read
from the item's recorded motion on the humanoid, in the frame of the window's first frame, by the position alone; where
the motion has ended before the window does, its last frame's goal. Early termination is off, so that every episode runs
its 8 s; whether CLoSD's own rule would have terminated it is recorded beside it.

Writes an .npz of body_pos (items, 270, 24, 3), the recorded second brought to 30 steps as CLoSD brings its plans and
the 240 control steps behind it, the bodies z up in the humanoid's order; fps; goals ("in"); and every item's length,
key and caption, in the order of the test items, with terminated (items,).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qinv, qrot
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_root_rot_pos
from closd.env.tasks import closd, closd_task
from closd.env.tasks.humanoid import compute_humanoid_reset
from closd.utils import torch_utils
from closd.utils.closd_util import STATES
from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

HISTORY, WINDOWS, FPS = 20, 4, 20  # the recorded second and the windows of 2 s, at HumanML3D's frame rate
GOAL_JOINT = {"Pelvis": "pelvis", "L_Wrist": "left_wrist", "R_Wrist": "right_wrist", "L_Toe": "left_foot", "R_Toe": "right_foot"}


def mjcf_bodies(mjcf):
    return list(ET.parse(mjcf).getroot().find("worldbody").iter("body"))


def start_states(qpos, orders, skeleton):
    """qpos[N,3,76] of mvp-prior2policy's humanoid at three frames (root quaternion XYZW; a body's three hinges in their
    `orders`, 'YXZ' turning about y first) -> CLoSD's state at the middle one, as PHC's motion library computes it: the
    bodies' positions and rotations, their velocities by finite differences with the other two frames, and each body's
    rotation against its parent and its rate as an exponential map."""
    count = len(qpos)
    angles = qpos[..., 7:].reshape(-1, len(orders), 3).numpy()
    local = [qpos[..., 3:7].reshape(-1, 4).numpy()] + [Rotation.from_euler(order, angles[:, b]).as_quat() for b, order in enumerate(orders)]
    local = torch.from_numpy(np.stack(local, 1)).float().view(count, 3, -1, 4)
    state = SkeletonState.from_rotation_and_root_translation(skeleton, local, qpos[..., :3].float(), is_local=True)
    pos, rot = state.global_translation, state.global_rotation
    vel = SkeletonMotion._compute_velocity(pos, 1 / FPS, guassian_filter=False)[:, 1]
    ang_vel = SkeletonMotion._compute_angular_velocity(rot, 1 / FPS, guassian_filter=False)[:, 1]
    angle, axis = torch_utils.quat_to_angle_axis(torch_utils.quat_mul(torch_utils.quat_conjugate(local[:, 1, 1:]), local[:, 2, 1:]))
    return dict(root_pos=pos[:, 1, 0], root_rot=rot[:, 1, 0], root_vel=vel[:, 0], root_ang_vel=ang_vel[:, 0],
                dof_pos=torch_utils.quat_to_exp_map(local[:, 1, 1:]).reshape(count, -1),
                dof_vel=(axis * angle[..., None] * FPS).reshape(count, -1),
                rigid_body_pos=pos[:, 1], rigid_body_rot=rot[:, 1], rigid_body_vel=vel, rigid_body_ang_vel=ang_vel)


class CLoSDGoals(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self.init_state = STATES.TEXT2MOTION  # a state without targets of its own
        root, self.out = Path(cfg["env"]["prior2policy"]), Path(cfg["env"]["out"])
        test = torch.load(root / "data/humanml3d_test.pt")
        goals = torch.load(root / "data/humanml3d_goals.pt")
        self.key, self.caption, self.length = np.array(goals["key"]), goals["caption"].numpy(), goals["length"].numpy()
        self.prompts = [test["captions"][i][c]["text"] for i, c in enumerate(self.caption.tolist())]
        count = len(self.prompts)
        recorded = torch.stack([motion[:HISTORY] for motion in test["motion"]]).float().to(self.device)
        self.prefix = (recorded - self.mean) / self.std  # [items, 20, 263], the first window's context
        self.history = self.rep.fps_convert(goals["body_pos"][:, :HISTORY].to(self.device), FPS, 30)  # [items, 30, 24, 3]
        bodies = mjcf_bodies(root / "assets/smpl_humanoid.xml")
        orders = ["".join("XYZ"[[float(a) for a in joint.get("axis").split()].index(1)] for joint in body.findall("joint")) for body in bodies[1:]]
        start = start_states(goals["qpos"][:, HISTORY - 2:HISTORY + 1], orders, self.skeleton_trees[0])
        self.start = {name: value.to(self.device) for name, value in start.items()}
        ended = goals["in_frame"] < 0  # a window's goal; where the motion has ended, its last frame's
        frame = torch.where(ended, goals["out_frame"][:, None], goals["in_frame"])
        body = torch.where(ended, goals["out_body"][:, None], goals["in_body"])
        self.goal_pos = goals["body_pos"][torch.arange(count)[:, None], frame, body].to(self.device)  # [items, 4, 3]
        self.goal_joint = [[GOAL_JOINT[bodies[b].get("name")] for b in item] for item in body.tolist()]

        self.item = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.round = 0
        self.bodies = torch.zeros(self.num_envs, self.max_episode_length - 1, *self._rigid_body_pos.shape[1:], device=self.device)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episodes = torch.zeros(count, self.history.shape[1] + self.bodies.shape[1], *self._rigid_body_pos.shape[1:])
        self.episode_terminated = torch.zeros(count, dtype=torch.bool)

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self.item[env_ids] = (self.round * self.num_envs + env_ids) % len(self.prompts)
            self.pose_buffer[env_ids] = self.history[self.item[env_ids]]
            self.terminated[env_ids] = False
        # CLoSD's own fills the context with the bodies as they are and replaces the first plan by them
        super(closd.CLoSD, self)._reset_envs(env_ids)

    def _reset_actors(self, env_ids):
        self._set_env_state(env_ids=env_ids, **{name: value[self.item[env_ids]] for name, value in self.start.items()})
        self._reset_ref_env_ids = env_ids  # the bodies' state set here is read until the simulator steps (humanoid_amp.py)

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)
        self._init_amp_obs_default(env_ids)

    def update_mdm_conditions(self, env_ids):
        for env, item in zip(env_ids.tolist(), self.item[env_ids].tolist()):
            self.hml_prompts[env] = self.prompts[item]

    def build_completion_input(self, context_switch_vec=None):
        window = (self.progress_buf // self.planning_horizon_30fps).clamp(max=WINDOWS - 1)
        rows = torch.arange(self.num_envs, device=self.device)
        context, goal, recon_data = self.rep.pose_to_hml(self.pose_buffer, self.goal_pos[self.item, window][:, None], fix_ik_bug=True)
        first = self.progress_buf < self.planning_horizon_30fps
        context[first] = self.prefix[self.item[first]]
        # the goal from the frame of the context's last frame, where the plan is placed, to that of the window's first frame
        quat, pos = recover_root_rot_pos(torch.cat([context, context[:, -1:]], 1) * self.std + self.mean)
        xz = torch.tensor([1., 0., 1.], device=self.device)
        goal = qrot(qinv(quat[:, -2]), goal[:, 0]) + pos[:, -2] * xz
        goal = qrot(quat[:, -1], goal - pos[:, -1] * xz)
        names = [self.goal_joint[item][k] for item, k in zip(self.item.tolist(), window.tolist())]
        target = torch.zeros(self.num_envs, self.num_joint_conditions, 3, device=self.device)
        target[rows, torch.tensor([self.extended_goal_joint_names.index(name) for name in names], device=self.device)] = goal
        aux_entries = dict(mask=torch.ones(self.num_envs, 1, 1, self.max_frame_20fps, dtype=torch.bool, device=self.mdm_device),
                           prefix_len=self.context_len_20fps, prefix=context.permute(0, 2, 1).unsqueeze(2),
                           scale=torch.full((self.num_envs,), self.mdm_cfg_param, device=self.mdm_device),
                           target_cond=target, target_joint_names=[[name] for name in names],
                           is_heading=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        return aux_entries, recon_data

    def post_physics_step(self):
        super().post_physics_step()
        step = int(self.progress_buf[0])  # the worlds step together
        self.bodies[:, step - 1] = self._rigid_body_pos
        self.terminated |= compute_humanoid_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, self._rigid_body_pos,
                                                  self.max_episode_length, True, self._termination_heights)[1].bool()
        if step == self.max_episode_length - 1:
            self.save_round()

    def save_round(self):
        count = len(self.prompts)
        kept = self.round * self.num_envs + torch.arange(self.num_envs, device=self.device) < count
        self.episodes[self.item[kept].cpu()] = torch.cat([self.history[self.item], self.bodies], 1)[kept].cpu()
        self.episode_terminated[self.item[kept].cpu()] = self.terminated[kept].cpu()
        self.round += 1
        print(f"{min(self.round * self.num_envs, count)} of {count} items", flush=True)
        if self.round * self.num_envs >= count:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.out, body_pos=self.episodes.numpy(), fps=30., goals="in", length=self.length, key=self.key,
                     caption=self.caption, terminated=self.episode_terminated.numpy())
            print(f"{count} episodes, {int(self.episode_terminated.sum())} that CLoSD would have terminated: {self.out}", flush=True)
            self.done_saving_episodes = True
