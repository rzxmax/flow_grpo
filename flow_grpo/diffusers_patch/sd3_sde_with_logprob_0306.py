# Adapted from flow matching to ODE deterministic mode
# Simplified version keeping only ODE flow

import math
from typing import Optional, Tuple
import torch


def ode_step_with_logprob(
    model_output: torch.FloatTensor,
    current_timestep_idx: int,
    sample: torch.FloatTensor,
    total_timesteps: int,
    prev_sample: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Flow-ODE单步更新（确定性模式）
    
    参数:
        model_output: 速度场v_t，shape同sample
        current_timestep_idx: timestep索引 [0, total_timesteps-1]
        sample: 当前状态x_t
        total_timesteps: 总timestep数（如10）
        prev_sample: 可选的预采样状态（用于训练时重算log_prob）
    
    返回: (next_sample, log_prob, sample_mean, std)
    
    参考: Flow matching ODE: x_t = (1-t)*x_0 + t*x_1, v_t = x_1 - x_0
    """
    device = sample.device
    dtype = sample.dtype
    
    # 1. 生成timesteps: t=0（噪声）→ t=1（数据）
    timesteps = torch.linspace(0, 1, total_timesteps + 1, device=device, dtype=dtype)
    
    # 2. 当前时间和增量
    t_input = timesteps[current_timestep_idx]
    delta = timesteps[current_timestep_idx + 1] - timesteps[current_timestep_idx]
    
    # 3. 扩展维度以匹配sample的shape
    if sample.dim() == 4:
        # 图像数据：(batch, channels, height, width)
        t_input = t_input.view(1, 1, 1, 1)
        delta = delta.view(1, 1, 1, 1)
    elif sample.dim() == 3:
        # 动作数据：(batch, action_horizon, action_dim)
        t_input = t_input.view(1, 1, 1)
        delta = delta.view(1, 1, 1)
    else:
        raise ValueError(f"Unsupported sample dimension: {sample.dim()}. Expected 3 or 4 dimensions.")
    
    # 4. 预测x0和x1
    #    Flow matching: x_t = (1-t)*x_0 + t*x_1, v_t = x_1 - x_0
    x0_pred = sample - model_output * t_input
    x1_pred = sample + model_output * (1 - t_input)
    
    # 5. 计算权重（ODE模式：无噪声）
    x0_weight = 1 - (t_input + delta)
    x1_weight = t_input + delta
    x_t_std = torch.zeros_like(t_input)
    
    # 6. 计算下一步状态（确定性）
    prev_sample_mean = x0_pred * x0_weight + x1_pred * x1_weight
    
    # 7. ODE模式：直接使用均值（无噪声）
    if prev_sample is None:
        prev_sample = prev_sample_mean
    # 如果prev_sample已提供（训练时重算log_prob），直接使用
    
    # 8. 计算log_prob（ODE模式下标准差为0，log_prob为0）
    log_prob = torch.zeros_like(sample)
    
    # 返回: (next_sample, log_prob, sample_mean, std)
    return prev_sample, log_prob, prev_sample_mean, x_t_std
