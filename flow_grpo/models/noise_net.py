"""
可学习的噪声网络（基于ReinFlow的ExploreNoiseNet实现）
用于预测状态和时间相关的噪声标准差
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """简单的MLP模块"""
    def __init__(self, dims: List[int], activation_type='Tanh', 
                 out_activation_type='Identity'):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if activation_type == 'Tanh':
                    layers.append(nn.Tanh())
                elif activation_type == 'ReLU':
                    layers.append(nn.ReLU())
                elif activation_type == 'Mish':
                    layers.append(nn.Mish())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ExploreNoiseNet(nn.Module):
    """
    可学习的噪声网络，基于时间和观测预测噪声标准差
    
    输入: [time_emb, obs_emb] - 时间编码和观测特征的拼接
    输出: noise_std - 噪声标准差，bounded in [min_std, max_std]
    
    参考ReinFlow的实现，通过策略梯度端到端训练
    """
    def __init__(
        self,
        in_dim: int,              # time_dim + obs_dim
        out_dim: int,             # action_horizon * action_dim
        noise_std_range: List[float],  # [min_std, max_std]
        device,
        hidden_dims: List[int] = [64, 64],
        activation_type: str = 'Tanh'
    ):
        super().__init__()
        self.device = device
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims + [out_dim],
            activation_type=activation_type,
            out_activation_type='Identity'
        ).to(self.device)
        
        self.set_noise_range(noise_std_range)
    
    def set_noise_range(self, noise_std_range: List[float]):
        """设置噪声范围 [min_std, max_std]"""
        min_std, max_std = noise_std_range
        self.logvar_min = nn.Parameter(
            torch.log(torch.tensor(min_std**2, dtype=torch.float32, device=self.device)),
            requires_grad=False
        )
        self.logvar_max = nn.Parameter(
            torch.log(torch.tensor(max_std**2, dtype=torch.float32, device=self.device)),
            requires_grad=False
        )
    
    def forward(self, noise_feature: torch.Tensor):
        """
        前向传播
        
        Args:
            noise_feature: (batch_size, in_dim) - concat([time_emb, obs_emb])
        
        Returns:
            noise_std: (batch_size, out_dim) - 每个动作元素的噪声标准差
        """
        noise_logvar = self.mlp_logvar(noise_feature)
        noise_logvar = torch.tanh(noise_logvar)
        # Map from [-1,1] to [logvar_min, logvar_max]
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std



