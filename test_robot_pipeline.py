#!/usr/bin/env python3
"""
测试脚本：验证robot_sde_pipeline_with_logprob_unet.py管线是否能正常运行
模拟maniskill环境的obs数据格式
"""

import sys
import os
sys.path.append('./flow_grpo')
sys.path.append('./flow_grpo/diffusers_patch')
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable

# 导入管线函数
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import pipeline_with_logprob, ActionDecoder, RobotActionPipeline


def create_test_data(batch_size=2, obs_dim=42, action_dim=8, action_horizon=16):
    """创建测试数据，模拟maniskill环境的obs格式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟不同episode的obs数据
    episodes_obs = []
    timesteps = [75, 50, 100]  # 不同episode的时间步长度
    
    for i in range(len(timesteps)):
        # 为每个episode创建obs: (timestep, obs_dim)
        episode_obs = torch.randn(timesteps[i], obs_dim)
        episodes_obs.append(episode_obs)
        print(f"Episode {i}: obs shape {episode_obs.shape}")
    
    # 为了测试，我们取每个episode的最后一个观测作为当前obs
    # 实际使用时，你可能需要根据具体需求处理时序obs
    current_obs = torch.stack([ep_obs[-1] for ep_obs in episodes_obs[:batch_size]])
    current_obs = current_obs.to(device)
    
    print(f"Current obs for pipeline: {current_obs.shape}")
    
    return current_obs, episodes_obs


def test_action_decoder():
    """测试ActionDecoder"""
    print("\n=== Testing ActionDecoder ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    action_dim = 8
    action_horizon = 16
    
    # 创建测试数据
    latents = torch.randn(batch_size, action_horizon, action_dim).to(device)
    
    # 创建decoder (使用正确的参数)
    decoder = ActionDecoder(
        action_dim=action_dim, 
        action_horizon=action_horizon,
        hidden_dim=256,
        num_layers=3
    ).to(device)
    
    # 测试解码
    actions = decoder(latents)
    
    print(f"✓ ActionDecoder test passed")
    print(f"  Input latents shape: {latents.shape}")
    print(f"  Output actions shape: {actions.shape}")
    print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    return decoder


def test_sde_vs_ode():
    """测试SDE vs ODE的区别 - 这是验证当前方法是SDE的核心测试"""
    print("\n=== Testing SDE vs ODE Behavior ===")
    
    # 参数设置
    batch_size = 1  # 使用单个样本以便比较
    obs_dim = 42
    action_dim = 8
    action_horizon = 16
    num_inference_steps = 10
    num_runs = 5  # 多次运行以测试随机性
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建固定的观测数据
    torch.manual_seed(42)  # 固定随机种子
    fixed_obs = torch.randn(batch_size, obs_dim).to(device)
    
    # 创建管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device
    )
    
    # 设置调度器
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipeline.setup_scheduler(FlowMatchEulerDiscreteScheduler())
    
    print("--- 测试1: 确定性模式 (determistic=True) ---")
    print("在确定性模式下，多次运行应该产生相同的结果")
    
    # 确定性模式下的多次运行
    deterministic_results = []
    for run in range(num_runs):
        torch.manual_seed(42)  # 每次都使用相同的种子
        result = pipeline.generate_actions(
            obs=fixed_obs,
            num_inference_steps=num_inference_steps,
            output_type="tensor",
            return_dict=True,
            determistic=True,  # 确定性模式
        )
        deterministic_results.append(result["actions"].cpu().numpy())
        print(f"  Run {run+1}: Mean action = {result['actions'].mean():.6f}")
    
    # 验证确定性模式的结果是否相同
    deterministic_identical = True
    for i in range(1, num_runs):
        if not np.allclose(deterministic_results[0], deterministic_results[i], atol=1e-6):
            deterministic_identical = False
            break
    
    print(f"✓ 确定性模式结果一致性: {'通过' if deterministic_identical else '失败'}")
    if deterministic_identical:
        print("  → 说明determistic=True时行为类似ODE（确定性）")
    
    print("\n--- 测试2: 随机模式 (determistic=False) ---")
    print("在随机模式下，多次运行应该产生不同的结果")
    
    # 随机模式下的多次运行
    stochastic_results = []
    for run in range(num_runs):
        torch.manual_seed(42 + run)  # 使用不同的种子
        result = pipeline.generate_actions(
            obs=fixed_obs,
            num_inference_steps=num_inference_steps,
            output_type="tensor",
            return_dict=True,
            determistic=False,  # 随机模式
        )
        stochastic_results.append(result["actions"].cpu().numpy())
        print(f"  Run {run+1}: Mean action = {result['actions'].mean():.6f}")
    
    # 验证随机模式的结果是否不同
    stochastic_different = False
    for i in range(1, num_runs):
        if not np.allclose(stochastic_results[0], stochastic_results[i], atol=1e-3):
            stochastic_different = True
            break
    
    print(f"✓ 随机模式结果差异性: {'通过' if stochastic_different else '失败'}")
    if stochastic_different:
        print("  → 说明determistic=False时行为类似SDE（随机性）")
    
    print("\n--- 测试3: 相同种子但不同确定性设置 ---")
    print("相同种子下，确定性和随机模式应该产生不同结果")
    
    # 相同种子下的确定性和随机模式
    torch.manual_seed(42)
    det_result = pipeline.generate_actions(
        obs=fixed_obs,
        num_inference_steps=num_inference_steps,
        output_type="tensor",
        return_dict=True,
        determistic=True,
    )
    
    torch.manual_seed(42)
    stoch_result = pipeline.generate_actions(
        obs=fixed_obs,
        num_inference_steps=num_inference_steps,
        output_type="tensor",
        return_dict=True,
        determistic=False,
    )
    
    det_stoch_different = not np.allclose(
        det_result["actions"].cpu().numpy(),
        stoch_result["actions"].cpu().numpy(),
        atol=1e-3
    )
    
    print(f"确定性 vs 随机模式 (相同种子): Mean diff = {abs(det_result['actions'].mean() - stoch_result['actions'].mean()):.6f}")
    print(f"✓ 确定性vs随机模式差异: {'通过' if det_stoch_different else '失败'}")
    
    print("\n--- 测试4: 分布统计分析 ---")
    print("分析多次运行的统计特性")
    
    # 收集更多样本进行统计分析
    num_samples = 20
    all_actions = []
    
    for i in range(num_samples):
        torch.manual_seed(i)
        result = pipeline.generate_actions(
            obs=fixed_obs,
            num_inference_steps=num_inference_steps,
            output_type="tensor",
            return_dict=True,
            determistic=False,
        )
        all_actions.append(result["actions"].cpu().numpy())
    
    all_actions = np.array(all_actions)  # (num_samples, batch_size, action_horizon, action_dim)
    
    # 统计分析
    mean_actions = np.mean(all_actions, axis=0)
    std_actions = np.std(all_actions, axis=0)
    
    print(f"  动作序列统计:")
    print(f"    平均值范围: [{mean_actions.min():.4f}, {mean_actions.max():.4f}]")
    print(f"    标准差范围: [{std_actions.min():.4f}, {std_actions.max():.4f}]")
    print(f"    平均标准差: {std_actions.mean():.4f}")
    
    # 验证是否有足够的随机性
    has_sufficient_randomness = std_actions.mean() > 0.01  # 阈值可调整
    print(f"✓ 随机性充足: {'通过' if has_sufficient_randomness else '失败'}")
    
    # 总结
    print("\n=== SDE验证总结 ===")
    sde_evidence = [
        ("确定性模式一致性", deterministic_identical),
        ("随机模式差异性", stochastic_different),
        ("确定性vs随机差异", det_stoch_different),
        ("随机性充足", has_sufficient_randomness)
    ]
    
    all_passed = all(passed for _, passed in sde_evidence)
    
    for test_name, passed in sde_evidence:
        print(f"  {test_name}: {'✓' if passed else '✗'}")
    
    if all_passed:
        print("\n🎉 所有测试通过！证明当前方法确实是SDE（随机微分方程）去噪过程")
        print("主要证据:")
        print("  1. determistic=True时表现出ODE特性（确定性）")
        print("  2. determistic=False时表现出SDE特性（随机性）")
        print("  3. 多次运行产生了有意义的随机变化")
        print("  4. 统计分析显示充足的随机性")
    else:
        print("\n❌ 部分测试失败，需要进一步检查SDE实现")
    
    return all_passed


def test_pipeline():
    """测试完整的pipeline"""
    print("\n=== Testing Robot Pipeline ===")
    
    # 参数设置
    batch_size = 2
    obs_dim = 42
    action_dim = 8
    action_horizon = 16
    num_inference_steps = 10  # 减少步数以加快测试
    
    # 创建测试数据
    current_obs, episodes_obs = create_test_data(batch_size, obs_dim, action_dim, action_horizon)
    
    # 创建管线（使用真正的RobotActionPipeline类）
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 设置调度器
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipeline.setup_scheduler(FlowMatchEulerDiscreteScheduler())
    
    print(f"\n--- Running pipeline inference ---")
    print(f"Device: {pipeline._execution_device}")
    print(f"Batch size: {batch_size}")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action horizon: {action_horizon}")
    print(f"Num inference steps: {num_inference_steps}")
    
    try:
        # 测试不同的output_type
        for output_type in ["tensor", "normalized", "clipped"]:
            print(f"\n--- Testing output_type: {output_type} ---")
            
            # 调用pipeline
            result = pipeline.generate_actions(
                obs=current_obs,
                num_inference_steps=num_inference_steps,
                output_type=output_type,
                return_dict=True,
                determistic=False,
            )
            
            # 验证输出
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert "actions" in result, "Missing 'actions' in result"
            assert "latents" in result, "Missing 'latents' in result"
            assert "log_probs" in result, "Missing 'log_probs' in result"
            assert "kl_divergences" in result, "Missing 'kl_divergences' in result"
            
            actions = result["actions"]
            latents = result["latents"]
            log_probs = result["log_probs"]
            kl_divs = result["kl_divergences"]
            
            print(f"✓ Pipeline test passed for output_type: {output_type}")
            print(f"  Actions shape: {actions.shape}")
            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"  Latents length: {len(latents)}")
            print(f"  Log probs length: {len(log_probs)}")
            print(f"  KL divergences length: {len(kl_divs)}")
            
            # 验证actions的范围
            if output_type == "normalized":
                assert actions.min() >= -1.0 and actions.max() <= 1.0, "Normalized actions should be in [-1, 1]"
            elif output_type == "clipped":
                assert actions.min() >= -1.0 and actions.max() <= 1.0, "Clipped actions should be in [-1, 1]"
        
        print(f"\n✓ All pipeline tests passed!")
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("Robot SDE Pipeline Test Script")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试组件
    success = True
    
    try:
        # 测试ActionDecoder
        test_action_decoder()
        
        # 测试SDE vs ODE特性 - 这是核心测试
        sde_success = test_sde_vs_ode()

        # 测试完整pipeline
        pipeline_success = test_pipeline()
        
        success = sde_success and pipeline_success
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The robot pipeline is working correctly.")
        print("✅ 确认当前方法是SDE（随机微分方程）去噪过程")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    main() 