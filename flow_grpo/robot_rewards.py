"""
机器人奖励评估模块

支持两种方式获取ManiSkill环境的奖励：
1. 实时运行环境评估 (maniskill_reward) - 需要训练好的模型
2. 从轨迹文件读取奖励 (maniskill_reward_from_trajectory) - 推荐方式

轨迹文件读取功能：
- 支持从 flow_maniskill_sde.py 生成的 HDF5 轨迹文件读取奖励
- 支持按时间戳筛选特定批次的轨迹
- 支持读取测试摘要文件
- 与原始评估函数接口完全兼容

使用示例：
1. 从轨迹文件读取奖励:
   scoring_fn = maniskill_reward_from_trajectory(
       trajectory_dir="./trajectory",
       env_id="PickCube-v1",
       timestamp_pattern="20250915_170017"  # 可选
   )
   scores, info = scoring_fn(obs, actions, metadata)

2. 实时环境评估（需要模型）:
   scoring_fn = maniskill_reward(
       action_unet=your_model, pipeline=your_pipeline, 
       action_decoder=your_decoder, ...
   )
   scores, info = scoring_fn(obs, actions, metadata)
"""

import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
import os
import collections
import h5py
import glob
from datetime import datetime
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import pipeline_with_logprob


def maniskill_reward(action_unet, pipeline, action_decoder, obs_dim, action_dim, action_horizon, 
                    env_id="PickCube-v1", env_kwargs=None, device="cuda", num_inference_steps=20, 
                    max_steps=500, n_test=3, test_start_seed=10000, deterministic=True, render=False):
    """
    在ManiSkill环境中评估机器人操作的奖励函数
    
    参数:
        action_unet: 训练好的UNet模型
        pipeline: RobotActionPipeline管线
        action_decoder: ActionDecoder解码器
        obs_dim: 观测维度
        action_dim: 动作维度
        action_horizon: 动作序列长度
        env_id: ManiSkill环境ID
        env_kwargs: 环境参数
        device: 计算设备
        num_inference_steps: 推理步数
        max_steps: 每个episode最大步数
        n_test: 测试次数
        test_start_seed: 测试起始种子
        deterministic: 是否使用确定性生成
        render: 是否渲染和保存视频
    
    返回:
        评估函数，接收(obs_batch, actions_batch, metadata)参数，返回(scores, info)
    """
    
    # 设置默认环境参数
    if env_kwargs is None:
        env_kwargs = {
            'sim_backend': 'physx_cpu',
            'control_mode': 'pd_ee_delta_pos',
            'obs_mode': 'state',
            'render_mode': 'rgb_array',
            'reward_mode': 'dense'
        }
    
    def _fn(obs_batch, actions_batch, metadata):
        """
        评估函数
        
        参数:
            obs_batch: 观测批次 (batch_size, obs_dim) - 当前暂不使用，直接在环境中重置
            actions_batch: 动作批次 (batch_size, action_horizon, action_dim) - 当前暂不使用，使用模型生成
            metadata: 元数据字典
        
        返回:
            scores: 奖励分数列表
            info: 额外信息字典
        """
        
        # 确保模型在正确的设备上且处于评估模式
        action_unet.to(device)
        action_decoder.to(device)
        action_unet.eval()
        action_decoder.eval()
        
        # 防止梯度计算
        for param in action_unet.parameters():
            param.requires_grad = False
        for param in action_decoder.parameters():
            param.requires_grad = False
        
        # 创建ManiSkill环境
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        
        # 如果需要渲染，设置录制
        if render:
            video_dir = "./videos"
            os.makedirs(video_dir, exist_ok=True)
            record_kwargs = {
                'save_trajectory': False,
                'save_video': True,
                'video_fps': 30
            }
            env = RecordEpisode(env, video_dir, **record_kwargs)
        
        # 运行多次测试
        all_rewards = []
        all_cumulative_rewards = []
        all_episode_lengths = []
        
        for test_idx in range(n_test):
            seed = test_start_seed + test_idx
            obs, info = env.reset(seed=seed)
            
            # 初始化观测张量
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(device, dtype=torch.float32)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            
            # 记录每步奖励和累计奖励
            episode_rewards = []
            done = False
            step_idx = 0
            
            while not done and step_idx < max_steps:
                # 使用模型预测动作序列
                with torch.no_grad():
                    result = pipeline_with_logprob(
                        pipeline,
                        obs=obs_tensor,
                        num_inference_steps=num_inference_steps,
                        output_type="clipped",  # 输出范围限制在[-1, 1]
                        return_dict=True,
                        determistic=deterministic,
                        action_decoder=action_decoder
                    )
                    
                    # 获取生成的动作序列
                    actions = result["actions"][0].cpu().numpy()  # 取第一个样本
                    
                    # 执行所有生成的动作
                    execute_horizon = action_horizon
                    for j in range(execute_horizon):
                        # 执行环境步骤
                        action = actions[j].copy()
                        
                        # 确保动作类型正确
                        if not isinstance(action, np.ndarray):
                            action = np.array(action, dtype=np.float32)
                        
                        try:
                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated
                        except Exception as e:
                            print(f"执行动作时出错: {e}")
                            done = True
                            reward = 0.0
                            break
                        
                        # 更新观测张量
                        if isinstance(obs, torch.Tensor):
                            obs_tensor = obs.to(device, dtype=torch.float32)
                            if obs_tensor.dim() == 1:
                                obs_tensor = obs_tensor.unsqueeze(0)
                        else:
                            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                        
                        # 记录奖励
                        reward_value = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                        episode_rewards.append(reward_value)
                        
                        # 更新步数
                        step_idx += 1
                        
                        # 检查是否结束
                        if done or step_idx >= max_steps:
                            done = True
                            break
            
            # 计算累计奖励
            cumulative_reward = sum(episode_rewards)
            all_rewards.extend(episode_rewards)
            all_cumulative_rewards.append(cumulative_reward)
            all_episode_lengths.append(len(episode_rewards))
            
            # 保存视频
            if render:
                env.flush_video(ignore_empty_transition=False)
        
        # 关闭环境
        env.close()
        
        # 计算统计信息
        avg_cumulative_reward = np.mean(all_cumulative_rewards)
        avg_episode_length = np.mean(all_episode_lengths)
        avg_step_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        # 计算平均奖励：轨迹总奖励除以轨迹长度
        # 这样可以标准化不同长度轨迹的奖励，更公平地比较不同episode的表现
        average_rewards = []
        for cumulative_reward, episode_length in zip(all_cumulative_rewards, all_episode_lengths):
            if episode_length > 0:
                average_reward = cumulative_reward / episode_length  # 每步平均奖励
            else:
                average_reward = 0.0  # 防止除零错误
            average_rewards.append(average_reward)
        
        # 返回平均奖励作为分数（而非累计奖励）
        scores = average_rewards
        
        # 构建额外信息
        info = {
            'avg_cumulative_reward': avg_cumulative_reward,
            'avg_episode_length': avg_episode_length,
            'avg_step_reward': avg_step_reward,
            'avg_average_reward': np.mean(average_rewards),  # 平均的平均奖励
            'all_cumulative_rewards': all_cumulative_rewards,  # 保留原始累计奖励信息
            'all_episode_rewards': all_rewards,
            'episode_lengths': all_episode_lengths,
            'n_episodes': n_test,
            'env_id': env_id,
            'total_steps': sum(all_episode_lengths)
        }
        
        return scores, info
    
    return _fn


def maniskill_reward_from_trajectory(trajectory_dir="./trajectory", env_id="PickCube-v1", 
                                    timestamp_pattern=None, max_episodes=None):
    """
    从已保存的轨迹文件中读取ManiSkill奖励信息
    
    参数:
        trajectory_dir: 轨迹文件目录路径
        env_id: ManiSkill环境ID，用于筛选轨迹文件
        timestamp_pattern: 时间戳模式筛选，例如 "20250915_170017"
        max_episodes: 最大读取的episode数量，None表示读取所有
    
    返回:
        评估函数，接收(obs_batch, actions_batch, metadata)参数，返回(scores, info)
    """
    
    def _fn(obs_batch, actions_batch, metadata):
        """
        从轨迹文件读取奖励评估函数
        
        参数:
            obs_batch: 观测批次 (当前未使用，保持接口一致性)
            actions_batch: 动作批次 (当前未使用，保持接口一致性) 
            metadata: 元数据字典
        
        返回:
            scores: 奖励分数列表
            info: 额外信息字典
        """
        
        print(f"从轨迹目录读取奖励: {trajectory_dir}")
        
        if not os.path.exists(trajectory_dir):
            print(f"❌ 轨迹目录不存在: {trajectory_dir}")
            return [], {'error': f'轨迹目录不存在: {trajectory_dir}'}
        
        # 构建文件搜索模式
        if timestamp_pattern:
            # 使用指定的timestamp模式
            file_pattern = os.path.join(trajectory_dir, f"episode_*_{env_id}_*_{timestamp_pattern}.hdf5")
        else:
            # 搜索所有该环境的轨迹文件
            file_pattern = os.path.join(trajectory_dir, f"episode_*_{env_id}_*.hdf5")
        
        # 查找匹配的轨迹文件
        trajectory_files = sorted(glob.glob(file_pattern))
        
        if not trajectory_files:
            print(f"❌ 未找到匹配的轨迹文件: {file_pattern}")
            return [], {'error': f'未找到匹配的轨迹文件: {file_pattern}'}
        
        # 限制读取的episode数量
        if max_episodes and len(trajectory_files) > max_episodes:
            trajectory_files = trajectory_files[:max_episodes]
        
        print(f"找到 {len(trajectory_files)} 个轨迹文件")
        
        # 读取轨迹数据
        all_rewards = []
        all_cumulative_rewards = []
        all_episode_lengths = []
        all_trajectories_info = []
        
        for traj_file in trajectory_files:
            try:
                # 读取HDF5轨迹文件
                with h5py.File(traj_file, 'r') as f:
                    # 读取基本数据
                    rewards = np.array(f['rewards'])
                    dones = np.array(f['dones'])
                    observations = np.array(f['observations'])
                    executed_actions = np.array(f['executed_actions'])
                    
                    # 读取episode信息
                    episode_info = {}
                    if 'episode_info' in f:
                        episode_info_group = f['episode_info']
                        for key in episode_info_group.attrs:
                            episode_info[key] = episode_info_group.attrs[key]
                    
                    # 计算统计信息
                    cumulative_reward = np.sum(rewards)
                    episode_length = len(rewards)
                    
                    # 存储结果
                    all_rewards.extend(rewards.tolist())
                    all_cumulative_rewards.append(float(cumulative_reward))
                    all_episode_lengths.append(episode_length)
                    
                    # 保存轨迹信息
                    traj_info = {
                        'file_path': traj_file,
                        'episode_info': episode_info,
                        'cumulative_reward': float(cumulative_reward),
                        'episode_length': episode_length,
                        'avg_reward': float(cumulative_reward / episode_length) if episode_length > 0 else 0.0,
                        'observations_shape': observations.shape,
                        'actions_shape': executed_actions.shape
                    }
                    all_trajectories_info.append(traj_info)
                    
                    print(f"✓ 读取轨迹: {os.path.basename(traj_file)}")
                    print(f"  Episode {episode_info.get('episode_id', 'N/A')}: {episode_length} 步, 累计奖励: {cumulative_reward:.3f}")
            
            except Exception as e:
                print(f"❌ 读取轨迹文件失败 {traj_file}: {e}")
                continue
        
        if not all_cumulative_rewards:
            return [], {'error': '未成功读取任何轨迹文件'}
        
        # 计算统计信息
        avg_cumulative_reward = np.mean(all_cumulative_rewards)
        avg_episode_length = np.mean(all_episode_lengths)
        avg_step_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        # 计算平均奖励：轨迹总奖励除以轨迹长度（与原函数保持一致）
        average_rewards = []
        for cumulative_reward, episode_length in zip(all_cumulative_rewards, all_episode_lengths):
            if episode_length > 0:
                average_reward = cumulative_reward / episode_length  # 每步平均奖励
            else:
                average_reward = 0.0  # 防止除零错误
            average_rewards.append(average_reward)
        
        # 返回平均奖励作为分数（与原函数保持一致）
        scores = average_rewards
        
        # 构建额外信息
        info = {
            'source': 'trajectory_files',
            'trajectory_dir': trajectory_dir,
            'timestamp_pattern': timestamp_pattern,
            'files_loaded': len(all_trajectories_info),
            'avg_cumulative_reward': avg_cumulative_reward,
            'avg_episode_length': avg_episode_length,
            'avg_step_reward': avg_step_reward,
            'avg_average_reward': np.mean(average_rewards),  # 平均的平均奖励
            'all_cumulative_rewards': all_cumulative_rewards,
            'all_episode_rewards': all_rewards,
            'episode_lengths': all_episode_lengths,
            'n_episodes': len(all_cumulative_rewards),
            'env_id': env_id,
            'total_steps': sum(all_episode_lengths),
            'trajectories_info': all_trajectories_info  # 详细轨迹信息
        }
        
        print(f"\n=== 轨迹奖励读取完成 ===")
        print(f"成功读取 {len(all_trajectories_info)} 个轨迹文件")
        print(f"平均累计奖励: {avg_cumulative_reward:.3f}")
        print(f"平均episode长度: {avg_episode_length:.1f}")
        print(f"平均每步奖励: {avg_step_reward:.3f}")
        print(f"平均的平均奖励: {np.mean(average_rewards):.3f}")
        
        return scores, info
    
    return _fn


def load_trajectory_summary(trajectory_dir="./trajectory", env_id="PickCube-v1", timestamp_pattern=None):
    """
    加载测试摘要文件
    
    参数:
        trajectory_dir: 轨迹文件目录路径
        env_id: ManiSkill环境ID
        timestamp_pattern: 时间戳模式筛选
    
    返回:
        summary_data: 摘要数据字典，如果未找到则返回None
    """
    
    # 构建摘要文件搜索模式
    if timestamp_pattern:
        summary_pattern = os.path.join(trajectory_dir, f"test_summary_{env_id}_{timestamp_pattern}.h5")
    else:
        summary_pattern = os.path.join(trajectory_dir, f"test_summary_{env_id}_*.h5")
    
    summary_files = glob.glob(summary_pattern)
    
    if not summary_files:
        print(f"未找到测试摘要文件: {summary_pattern}")
        return None
    
    # 使用最新的摘要文件
    summary_file = sorted(summary_files)[-1]
    print(f"读取测试摘要: {summary_file}")
    
    try:
        with h5py.File(summary_file, 'r') as f:
            summary_data = {}
            
            # 读取数组数据
            if 'all_rewards' in f:
                summary_data['all_rewards'] = [np.array(r) for r in f['all_rewards']]
            if 'total_rewards_per_episode' in f:
                summary_data['total_rewards_per_episode'] = np.array(f['total_rewards_per_episode'])
            if 'episode_lengths' in f:
                summary_data['episode_lengths'] = np.array(f['episode_lengths'])
            if 'avg_rewards_per_episode' in f:
                summary_data['avg_rewards_per_episode'] = np.array(f['avg_rewards_per_episode'])
            
            # 读取测试摘要信息
            if 'test_summary' in f:
                test_summary = {}
                summary_group = f['test_summary']
                for key in summary_group.attrs:
                    test_summary[key] = summary_group.attrs[key]
                summary_data['test_summary'] = test_summary
                
                print(f"✓ 摘要信息:")
                print(f"  环境: {test_summary.get('env_id', 'N/A')}")
                print(f"  测试次数: {test_summary.get('n_test', 'N/A')}")
                print(f"  平均累计奖励: {test_summary.get('avg_total_reward', 'N/A'):.3f}")
                print(f"  平均episode长度: {test_summary.get('avg_episode_length', 'N/A'):.1f}")
            
            return summary_data
            
    except Exception as e:
        print(f"读取测试摘要失败: {e}")
        return None


# 后续再补充多种奖励来源

def main():
    """
    测试函数 - 演示从轨迹文件读取奖励的功能
    """
    print("=== Robot Rewards 测试 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 方式1: 从轨迹文件读取奖励（推荐）
    print("\n--- 从轨迹文件读取奖励 ---")
    
    # 创建从轨迹读取奖励的评估函数
    trajectory_scoring_fn = maniskill_reward_from_trajectory(
        trajectory_dir="./trajectory",  # 轨迹文件目录
        env_id="PickCube-v1",          # 环境ID
        timestamp_pattern="20250915_170017",  # 指定时间戳（可选）
        max_episodes=5                 # 最大读取episode数量（可选）
    )
    
    # 调用评估函数（这里obs和actions不会被使用，仅为了保持接口一致性）
    dummy_obs = torch.randn(1, 42)  # 假设obs_dim=42
    dummy_actions = torch.randn(1, 16, 8)  # 假设action_horizon=16, action_dim=8
    metadata = {}
    
    try:
        scores, info = trajectory_scoring_fn(dummy_obs, dummy_actions, metadata)
        
        print(f"\n✓ 轨迹奖励读取成功!")
        print(f"  读取的episode数量: {info.get('n_episodes', 0)}")
        print(f"  平均累计奖励: {info.get('avg_cumulative_reward', 0):.3f}")
        print(f"  平均episode长度: {info.get('avg_episode_length', 0):.1f}")
        print(f"  分数 (每步平均奖励): {scores}")
        
        # 显示每个轨迹的详细信息
        if 'trajectories_info' in info:
            print(f"\n--- 轨迹详细信息 ---")
            for i, traj_info in enumerate(info['trajectories_info']):
                print(f"  轨迹 {i+1}: {os.path.basename(traj_info['file_path'])}")
                print(f"    累计奖励: {traj_info['cumulative_reward']:.3f}")
                print(f"    平均奖励: {traj_info['avg_reward']:.3f}")
                print(f"    步数: {traj_info['episode_length']}")
    
    except Exception as e:
        print(f"❌ 轨迹奖励读取失败: {e}")
    
    # 方式2: 读取测试摘要文件
    print(f"\n--- 读取测试摘要文件 ---")
    
    try:
        summary_data = load_trajectory_summary(
            trajectory_dir="./trajectory",
            env_id="PickCube-v1",
            timestamp_pattern="20250915_170017"  # 可选
        )
        
        if summary_data:
            test_summary = summary_data.get('test_summary', {})
            print(f"✓ 测试摘要读取成功!")
            print(f"  测试时间: {test_summary.get('timestamp', 'N/A')}")
            print(f"  检查点: {test_summary.get('checkpoint_path', 'N/A')}")
            print(f"  确定性模式: {test_summary.get('deterministic', 'N/A')}")
    
    except Exception as e:
        print(f"❌ 测试摘要读取失败: {e}")
    
    # 方式3: 原始方式 - 直接在环境中运行（需要模型）
    print(f"\n--- 原始方式（需要真实模型，这里仅作演示）---")
    print("如果您有训练好的模型，可以使用 maniskill_reward() 函数:")
    print("scoring_fn = maniskill_reward(")
    print("    action_unet=your_unet,")
    print("    pipeline=your_pipeline,")
    print("    action_decoder=your_decoder,")
    print("    obs_dim=42, action_dim=8, action_horizon=16,")
    print("    env_id='PickCube-v1', device=device")
    print(")")
    print("scores, info = scoring_fn(obs, actions, metadata)")


def demo_trajectory_analysis():
    """
    演示轨迹分析功能
    """
    print("\n=== 轨迹分析演示 ===")
    
    # 分析轨迹目录
    trajectory_dir = "./trajectory"
    
    if not os.path.exists(trajectory_dir):
        print(f"❌ 轨迹目录不存在: {trajectory_dir}")
        print("请先运行 flow_maniskill_sde.py --test 生成轨迹文件")
        return
    
    # 列出所有轨迹文件
    all_hdf5_files = glob.glob(os.path.join(trajectory_dir, "*.hdf5"))
    episode_files = [f for f in all_hdf5_files if os.path.basename(f).startswith('episode_')]
    summary_files = [f for f in all_hdf5_files if os.path.basename(f).startswith('test_summary_')]
    
    print(f"轨迹目录: {trajectory_dir}")
    print(f"找到 {len(episode_files)} 个episode文件")
    print(f"找到 {len(summary_files)} 个摘要文件")
    
    # 按环境分组分析
    env_groups = {}
    for file_path in episode_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) >= 3:
            env_id = parts[2]  # episode_{idx}_{env_id}_{seed}_{timestamp}.hdf5
            if env_id not in env_groups:
                env_groups[env_id] = []
            env_groups[env_id].append(file_path)
    
    # 分析每个环境的轨迹
    for env_id, files in env_groups.items():
        print(f"\n--- 环境 {env_id} ---")
        print(f"轨迹数量: {len(files)}")
        
        # 使用轨迹读取函数分析
        scoring_fn = maniskill_reward_from_trajectory(
            trajectory_dir=trajectory_dir,
            env_id=env_id
        )
        
        dummy_obs = torch.randn(1, 42)
        dummy_actions = torch.randn(1, 16, 8) 
        scores, info = scoring_fn(dummy_obs, dummy_actions, {})
        
        if info.get('n_episodes', 0) > 0:
            print(f"平均累计奖励: {info.get('avg_cumulative_reward', 0):.3f}")
            print(f"最高累计奖励: {max(info.get('all_cumulative_rewards', [0])):.3f}")
            print(f"最低累计奖励: {min(info.get('all_cumulative_rewards', [0])):.3f}")
            print(f"成功率估计: {len([r for r in info.get('all_cumulative_rewards', []) if r > 50])} / {info.get('n_episodes', 0)}")


if __name__ == "__main__":
    # 运行主要测试
    main()
    
    # 运行轨迹分析演示
    demo_trajectory_analysis()
