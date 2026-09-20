# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `ode_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
# - Simplified to only use ODE deterministic mode
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps, XLA_AVAILABLE
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from .sd3_sde_with_logprob import ode_step_with_logprob



import sys
from pathlib import Path
sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from models.unet import ConditionalUnet1D
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from termcolor import colored
import pathlib
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
import h5py


class RobotActionPipeline:
    """
    机器人动作生成管线
    
    这个类展示了如何将 ActionDecoder 集成到机器人控制系统中
    """
    
    def __init__(
        self,
        obs_dim: int = 42,
        action_dim: int = 4,
        action_horizon: int = 16,
        device: str = "cuda"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 初始化 ActionDecoder (使用正确的参数)
        self.action_decoder = ActionDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=256,
            num_layers=3
        ).to(self.device)
        
        # 其他必要的组件
        self.scheduler = None
        self._execution_device = self.device
        self._action_unet = None
        self._interrupt = False
        
    def setup_scheduler(self, scheduler):
        """设置调度器"""
        self.scheduler = scheduler
        
    def check_inputs(self, obs, height, width, **kwargs):
        """输入验证"""
        if obs is not None:
            assert isinstance(obs, torch.Tensor), "obs must be a torch.Tensor"
            assert obs.dim() == 2, f"obs must be 2D (batch_size, obs_dim), got {obs.shape}"
            assert obs.shape[1] == self.obs_dim, f"obs_dim mismatch: expected {self.obs_dim}, got {obs.shape[1]}"
        
        # print(f"✓ Input validation passed")
        # print(f"  obs shape: {obs.shape if obs is not None else None}")
        # print(f"  height: {height}, width: {width}")
        
    def maybe_free_model_hooks(self):
        """释放模型钩子"""
        pass
        
    def progress_bar(self, total):
        # """进度条"""
        # return tqdm(total=total, desc="Denoising")
        """进度条（已禁用）"""
        # 返回一个模拟的进度条对象，update 方法不执行任何操作
        class DummyProgressBar:
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
            def update(self, n=1):
                pass
                
        return DummyProgressBar()
        
    @property
    def interrupt(self):
        """中断标志"""
        return self._interrupt
        
    def generate_actions(
        self,
        obs: torch.Tensor,
        num_inference_steps: int = 28,
        output_type: str = "tensor",
        return_dict: bool = True,
        **kwargs
    ):
        """
        生成机器人动作序列（ODE确定性模式）
        
        Args:
            obs: 观测数据 (batch_size, obs_dim)
            num_inference_steps: 推理步数
            output_type: 输出类型 ("tensor", "normalized", "clipped")
            return_dict: 是否返回字典格式
            
        Returns:
            actions: 动作序列或包含动作的字典
        """
        # 确保使用类的action_decoder，除非明确指定了其他decoder
        action_decoder = kwargs.pop('action_decoder', self.action_decoder)
        
        return pipeline_with_logprob(
            self,
            obs=obs,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
            return_dict=return_dict,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            action_decoder=action_decoder,
            **kwargs
        )

class ActionDecoder(nn.Module):
    """
    MLP解码器，将latents解码为动作序列
    """
    def __init__(self, action_dim=4, action_horizon=16, hidden_dim=256, num_layers=3):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # 输入维度：action_horizon * action_dim
        input_dim = action_horizon * action_dim
        
        # 构建MLP层
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        # 最后一层输出动作序列
        layers.append(nn.Linear(current_dim, input_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 可选的标准化参数
        self.register_buffer('scaling_factor', torch.tensor(1.0))
        self.register_buffer('shift_factor', torch.tensor(0.0))
        
    def forward(self, latents):
        """
        Args:
            latents: (batch_size, action_horizon, action_dim)
        Returns:
            actions: (batch_size, action_horizon, action_dim)
        """
        batch_size = latents.shape[0]
        
        # 展平latents（使用reshape处理非连续内存）
        latents_flat = latents.reshape(batch_size, -1)
        
        # 通过MLP解码
        actions_flat = self.mlp(latents_flat)
        
        # 重塑回原始维度（使用reshape处理非连续内存）
        actions = actions_flat.reshape(batch_size, self.action_horizon, self.action_dim)
        
        return actions


# @torch.no_grad()
def pipeline_with_logprob(
    self,
    obs: Union[torch.Tensor] = None,
    height: Optional[int] = 128,
    width: Optional[int] = 128,
    num_inference_steps: int = 28, # 去噪步骤的数量。更多的去噪步骤通常会产生更高质量的动作序列，但推理速度会变慢
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # 用于使生成过程确定性的随机数生成器
    latents: Optional[torch.FloatTensor] = None, # 预生成的噪声潜在变量，从高斯分布采样，用作动作生成的输入
    obs_embeds: Optional[torch.FloatTensor] = None, # 预处理过的 obs 嵌入
    pooled_obs_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "tensor",
    return_dict: bool = True,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None, # 用于让用户深度介入扩散模型生成过程的参数
    action_dim: int = 4,
    action_horizon: int = 16,
    action_decoder: Optional[ActionDecoder] = None,
    kl_reward: float = 0.0,
):

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        obs,
        height,
        width,
        obs_embeds=obs_embeds,
        pooled_obs_embeds=pooled_obs_embeds,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    self._interrupt = False

    # 2. Define call parameters
    if obs is not None and isinstance(obs, torch.Tensor):
        batch_size = obs.shape[0]
    else:
        batch_size = 1
    
    device = self._execution_device

    # lora_scale = (

    # 3. 初始化 ConditionalUnet1D（应该在循环外初始化）
    if not hasattr(self, '_action_unet') or self._action_unet is None:
        self._action_unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs.shape[1] if obs is not None else 128
        ).to(device)
        
    # TODO：obs_embeds，对输入obs进行编码，应该在图像的时候需要，状态向量先不处理
    # num_channels_latents = self.transformer.config.in_channels

    # 4. 编码观测数据（如果需要）
    if obs_embeds is None and obs is not None:
        # 这里需要根据实际情况编码obs
        # 暂时直接使用obs作为条件
        obs_embeds = obs
    
    # TODO：latents，从高斯分布采样
    # 图像生成中先用 prepare_latents 函数生成，方便后续处理
    # 这里 unet 尝试直接处理噪声 action

    # ✅ DiffusionNFT风格：使用 generator 生成可控的随机噪声
    # 参考 DiffusionNFT/flow_grpo/diffusers_patch/pipeline_with_logprob.py Line 154-163
    # 
    # 方案A：支持批量generator（列表形式）
    # - 当 generator 是列表时，为每个样本单独生成初始噪声
    # - 当 generator 是单个时，保持原有行为（所有样本共享）
    if latents is None:
        if generator is not None:
            # 检查是否是列表（批量generator）
            if isinstance(generator, list):
                # 批量模式：每个样本单独生成latents
                if len(generator) != batch_size:
                    raise ValueError(
                        f"generator列表长度({len(generator)})必须等于batch_size({batch_size})"
                    )
                
                # 为每个样本单独生成初始噪声
                latents_list = []
                for i in range(batch_size):
                    latent_i = torch.randn(
                        1, action_horizon, action_dim,
                        generator=generator[i],
                        device=device,
                        dtype=torch.float32
                    )
                    latents_list.append(latent_i)
                
                latents = torch.cat(latents_list, dim=0)
            else:
                # 单个generator：所有样本共享（原行为）
                latents = torch.randn(
                    batch_size, action_horizon, action_dim,
                    generator=generator,
                    device=device,
                    dtype=torch.float32
                )
        else:
            # 如果没有传入 generator，使用全局随机状态（原行为）
            latents = torch.randn(batch_size, action_horizon, action_dim).to(device)
    else:
        latents = latents.to(device)

    # 6. 不再使用scheduler，timesteps由循环索引生成
    self._num_timesteps = num_inference_steps

    all_latents = [latents]
    all_log_probs = []
    all_kl = []

    # 7. Denoising loop（ODE确定性模式）
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i in range(num_inference_steps):
            if self.interrupt:
                continue
            
            # 使用UNet预测速度场
            # 确保obs_embeds的维度正确
            if obs_embeds is not None:
                noise_pred_raw = self._action_unet(
                    latents,  # 当前latents
                    timestep=i / num_inference_steps,  # 归一化到[0,1]
                    global_cond=obs_embeds  # 观测条件
                )
            else:
                # 如果没有观测条件，使用零向量
                print("no obs_embeds")
                zero_cond = torch.zeros(batch_size, 128).to(device)
                noise_pred_raw = self._action_unet(
                    latents,  # 当前latents
                    timestep=i / num_inference_steps,  # 归一化到[0,1]
                    global_cond=zero_cond  # 观测条件
                )
            
            # 将UNet和Decoder作为CompleteModel = Decoder ∘ UNet
            if action_decoder is not None:
                noise_pred = action_decoder(noise_pred_raw)
            else:
                noise_pred = noise_pred_raw

            # 调用ODE步骤（确定性模式）
            latents, log_prob, prev_latents_mean, std_dev_t = ode_step_with_logprob(
                noise_pred.float(),           # model_output（速度场预测）
                current_timestep_idx=i,       # 当前timestep索引
                sample=latents.float(),       # 当前状态
                total_timesteps=num_inference_steps,  # 总timestep数
            )
                
            prev_latents = latents.clone()

            all_latents.append(latents)
            all_log_probs.append(log_prob)
            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)
            # if callback_on_step_end is not None:

            # TODO：KL奖励计算（如果需要）

            # 更新进度条
            progress_bar.update()

    if output_type == "latent":
        actions = latents
        self.maybe_free_model_hooks()
        return actions
    else:
        # ===== 关键：latents已经是CompleteModel处理后的结果 =====
        # 在ODE循环中，noise_pred已经通过action_decoder
        # 因此latents已经在"decoder处理后的空间"中
        # 直接作为actions，不需要再解码
        actions = latents
        
        # 可选的后处理步骤，例如动作范围限制
        if output_type == "normalized":
            # 将动作归一化到 [-1, 1] 范围
            actions = torch.tanh(actions)
        elif output_type == "clipped":
            # 将动作限制在 [-1, 1] 范围
            actions = torch.clamp(actions, -1, 1)
        # 其他情况保持原始输出

    # ========== Pipeline输出验证（已禁用） ==========
    # 注：log_prob警告已被禁用，避免输出过多日志
    # 如需调试，可以取消注释以下代码：
    # for t, lp in enumerate(all_log_probs):
    #     if torch.isnan(lp).any():
    #         print(f"⚠️  Pipeline警告: timestep {t} 的log_prob包含NaN!")
    #     if torch.isinf(lp).any():
    #         print(f"⚠️  Pipeline警告: timestep {t} 的log_prob包含Inf!")
    #     if (lp > 0).any():
    #         positive_count = (lp > 0).sum().item()
    #         max_val = lp.max().item()
    #         print(f"⚠️  Pipeline警告: timestep {t} 的log_prob包含 {positive_count} 个正值! 最大值: {max_val:.6f}")
    
    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (actions, all_latents, all_log_probs, all_kl)

    # 返回类似StableDiffusion3PipelineOutput的格式，但包含动作而不是图像
    return {
        "actions": actions,
        "latents": all_latents,
        "log_probs": all_log_probs,
        "kl_divergences": all_kl
    }
    
    
