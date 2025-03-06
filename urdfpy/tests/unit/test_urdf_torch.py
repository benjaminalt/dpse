import torch
import numpy as np
import matplotlib.pyplot as plt

from urdfpy import URDF
from urdfpy.utils import show_joint_trajectory


def test_link_fk_torch():
    u = URDF.load('tests/data/ur5/ur5.urdf')
    fk_torch = u.link_fk_torch(torch.zeros(6, dtype=torch.float32))
    fk = u.link_fk(np.zeros(6))
    fk_list = list(fk.items())
    fk_torch_list = list(fk_torch.items())
    for i in range(len(fk)):
        tf_orig = fk_list[i][1]
        tf_torch = fk_torch_list[i][1]
        assert tf_torch.allclose(torch.tensor(tf_orig, dtype=torch.float32))


def test_link_fk_batch_torch():
    u = URDF.load('tests/data/ur5/ur5.urdf')
    fk_torch = u.link_fk_batch_torch(torch.zeros([1, 6], dtype=torch.float32))
    fk = u.link_fk(np.zeros(6))
    fk_list = list(fk.items())
    fk_torch_list = list(fk_torch.items())
    for i in range(len(fk)):
        tf_orig = fk_list[i][1]
        tf_torch = fk_torch_list[i][1]
        assert tf_torch.allclose(torch.tensor(tf_orig, dtype=torch.float32))


def test_link_fk_torch_grad():
    u = URDF.load('tests/data/ur5/ur5.urdf')
    joint_states = torch.zeros(6, dtype=torch.float32, requires_grad=True)
    fk_torch = u.link_fk_torch(joint_states)
    for link, tf in fk_torch.items():
        if link.name == "ee_link":
            torch.nn.MSELoss()(tf, torch.zeros_like(tf)).backward()
            assert joint_states.grad is not None
            assert not torch.allclose(joint_states.grad, torch.zeros_like(joint_states.grad))


def test_optimize_input_joint_states_ur5():
    u = URDF.load('tests/data/ur5/ur5.urdf')
    joint_states = torch.zeros(6, dtype=torch.float32, requires_grad=True)
    ee_link_pose_goal = torch.tensor([[8.7428e-08, -1.0000e+00, -9.7932e-12, -8.1725e-01],
                                      [-1.0000e+00, -8.7428e-08, -8.5615e-19, -1.9145e-01],
                                      [-4.7953e-23, 9.7932e-12, -1.0000e+00, -5.4910e-03],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]], dtype=torch.float32)
    optim = torch.optim.Adam([joint_states], lr=1e-3)
    losses = []
    joint_values = []
    for i in range(10000):
        optim.zero_grad()
        fk_torch = u.link_fk_torch(joint_states)
        for link, tf in fk_torch.items():
            if link.name == "ee_link":
                loss = torch.nn.MSELoss()(tf, ee_link_pose_goal)
                loss.backward()
                optim.step()
                print(f"{i}: Loss {loss.item()}, {joint_states.tolist()}")
                losses.append(loss.item())
                joint_values.append(joint_states.tolist())
                break
    joint_values = torch.tensor(joint_values)
    show_joint_trajectory(u, joint_values)


def test_optimize_input_joint_states_ur5_batch():
    u = URDF.load('tests/data/ur5/ur5.urdf')
    joint_states = torch.zeros([1, 6], dtype=torch.float32, requires_grad=True)
    ee_link_pose_goal = torch.tensor([[[8.7428e-08, -1.0000e+00, -9.7932e-12, -8.1725e-01],
                                      [-1.0000e+00, -8.7428e-08, -8.5615e-19, -1.9145e-01],
                                      [-4.7953e-23, 9.7932e-12, -1.0000e+00, -5.4910e-03],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]], dtype=torch.float32)
    optim = torch.optim.Adam([joint_states], lr=1e-3)
    losses = []
    joint_values = []
    for i in range(10000):
        optim.zero_grad()
        fk_torch = u.link_fk_batch_torch(joint_states)
        for link, tf in fk_torch.items():
            if link.name == "ee_link":
                loss = torch.nn.MSELoss()(tf, ee_link_pose_goal)
                loss.backward()
                optim.step()
                print(f"{i}: Loss {loss.item()}, {joint_states.tolist()}")
                losses.append(loss.item())
                joint_values.append(joint_states.tolist())
                break
    joint_values = torch.tensor(joint_values).squeeze()
    show_joint_trajectory(u, joint_values)


def test_optimize_input_joint_states_with_gripper():
    u = URDF.load('tests/data/ur5_robotiq.urdf')
    joint_states = torch.zeros(7, dtype=torch.float32, requires_grad=True)
    ee_link_pose_goal = torch.tensor([[8.7428e-08, -1.0000e+00, -9.7932e-12, -8.1725e-01],
                                      [-1.0000e+00, -8.7428e-08, -8.5615e-19, -1.9145e-01],
                                      [-4.7953e-23, 9.7932e-12, -1.0000e+00, -5.4910e-03],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]], dtype=torch.float32)
    optim = torch.optim.Adam([joint_states], lr=1e-3)
    losses = []
    joint_values = []
    for i in range(5000):
        optim.zero_grad()
        tf = u.link_fk_torch(joint_states, "ee_link")
        loss = torch.nn.MSELoss()(tf, ee_link_pose_goal)
        loss.backward()
        optim.step()
        print(f"{i}: Loss {loss.item()}, {joint_states.tolist()}")
        losses.append(loss.item())
        joint_values.append(joint_states.tolist())
    joint_values = torch.tensor(joint_values)
    show_joint_trajectory(u, joint_values)


def test_optimize_input_joint_states_with_gripper_batch():
    u = URDF.load('tests/data/ur5_robotiq.urdf')
    joint_states = torch.zeros([1, 7], dtype=torch.float32, requires_grad=True)
    ee_link_pose_goal = torch.tensor([[[8.7428e-08, -1.0000e+00, -9.7932e-12, -8.1725e-01],
                                      [-1.0000e+00, -8.7428e-08, -8.5615e-19, -1.9145e-01],
                                      [-4.7953e-23, 9.7932e-12, -1.0000e+00, -5.4910e-03],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]], dtype=torch.float32)
    optim = torch.optim.Adam([joint_states], lr=1e-3)
    losses = []
    joint_values = []
    for i in range(5000):
        optim.zero_grad()
        tf = u.link_fk_batch_torch(joint_states, "ee_link")
        loss = torch.nn.MSELoss()(tf, ee_link_pose_goal)
        loss.backward()
        optim.step()
        print(f"{i}: Loss {loss.item()}, {joint_states.tolist()}")
        losses.append(loss.item())
        joint_values.append(joint_states.tolist())
    joint_values = torch.tensor(joint_values).squeeze()
    show_joint_trajectory(u, joint_values)


if __name__ == '__main__':
    test_optimize_input_joint_states_with_gripper()
