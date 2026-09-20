import math
import torch
from typing import Optional, Tuple, Union

def ode_step(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = True,  # ODE默认是确定性的
) -> Tuple:
    """
    使用ODE(常微分方程)进行一步反向扩散，无需添加随机噪声
    
    参数:
        scheduler: 扩散调度器
        model_output: 模型预测的噪声或速度场 (batch_size, action_horizon, action_dim)
        timestep: 当前时间步
        sample: 当前状态 (batch_size, action_horizon, action_dim)
        prev_sample: 如果提供，则使用这个作为下一个状态，而不是计算
        generator: 随机数生成器 (在ODE中不使用)
        determistic: 是否使用确定性生成 (ODE中通常为True)
    
    返回:
        prev_sample: 更新后的状态
        log_prob: 对数概率 (用于训练或评估)
        prev_sample_mean: 更新后状态的均值
        std_dev: 标准差 (在ODE中设为0)
    """
    # 获取时间步索引
    # 注意：ODE使用离散步骤，因此我们需要找到最接近当前时间步的索引
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(scheduler.timesteps.device)
    
    # 计算步长和相邻时间步
    if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
        # 找到当前时间步在scheduler中的索引位置
        indices = (scheduler.timesteps - timestep.cpu()).abs().argmin().item()
        indices = [indices]
        
        # 计算前一个时间步索引
        prev_indices = [min(i + 1, len(scheduler.timesteps) - 1) for i in indices]
    else:
        # 如果调度器没有timesteps属性，创建一个线性时间序列
        step_ratio = 0.1  # 默认步长
        indices = [0]
        prev_indices = [1]

    # 在ODE流程中，我们直接使用当前状态和模型输出计算下一个状态
    # 根据Euler方法: x_{t-dt} = x_t + v_t * dt
    # 其中v_t是模型预测的速度场
    
    # 计算时间步长
    if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None:
        # 对于使用sigma的调度器
        sigma = scheduler.sigmas[indices].to(sample.device)
        sigma_prev = scheduler.sigmas[prev_indices].to(sample.device)
        
        # 根据sample的维度调整sigma的形状
        if sample.dim() == 4:
            # 图像数据：(batch_size, channels, height, width)
            sigma = sigma.view(-1, 1, 1, 1)
            sigma_prev = sigma_prev.view(-1, 1, 1, 1)
        elif sample.dim() == 3:
            # 机器人动作数据：(batch_size, action_horizon, action_dim)
            sigma = sigma.view(-1, 1, 1)
            sigma_prev = sigma_prev.view(-1, 1, 1)
        else:
            raise ValueError(f"Unsupported sample dimension: {sample.dim()}. Expected 3 or 4 dimensions.")
        
        dt = sigma_prev - sigma
    else:
        # 对于没有sigma的调度器，使用固定步长
        dt = 0.1
    
    # 在ODE中，我们直接使用模型输出和时间步长计算下一个状态
    # 不添加随机噪声，完全确定性
    prev_sample_mean = sample + dt * model_output
    
    # 如果提供了prev_sample，使用它；否则使用我们计算的mean
    if prev_sample is None:
        prev_sample = prev_sample_mean
    
    # 在ODE流程中，由于我们不添加噪声，log_prob需要另外计算
    # 我们可以假设一个小的固定方差来计算log_prob
    small_sigma = 1e-5
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * small_sigma**2)
        - torch.log(torch.tensor(small_sigma, device=prev_sample.device))
        - torch.log(torch.sqrt(torch.tensor(2 * math.pi, device=prev_sample.device)))
    )
    
    # 沿着所有非批次维度取平均
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # 返回计算出的状态、对数概率、均值和标准差（ODE中为0）
    return prev_sample, log_prob, prev_sample_mean, torch.tensor(small_sigma, device=prev_sample.device) 