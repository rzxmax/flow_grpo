# 机器人操作管线 - 从图像/状态生成动作
# 基于 sd3_pipeline_with_logprob.py 修改

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from .sd3_sde_with_logprob import sde_step_with_logprob
from PIL import Image
import numpy as np

class RobotStateEncoder(nn.Module):
    """将输入图像或状态向量编码为条件嵌入 - 支持动态状态维度"""
    def __init__(self, image_input_dim=3, state_input_dim=None, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.state_input_dim = state_input_dim
        self.hidden_dim = hidden_dim
        
        # 图像编码分支
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(image_input_dim, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 状态向量编码分支 - 支持动态输入维度
        self.state_encoder = None
        if state_input_dim is not None:
            self._build_state_encoder(state_input_dim)
        
        # 最终映射层
        if state_input_dim is None:
            # 仅支持图像输入
            fc_input_dim = 256 * 8 * 8
        else:
            # 支持状态向量输入
            fc_input_dim = hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 如果同时支持图像和状态向量，需要分别的映射层
        if state_input_dim is not None:
            self.image_to_hidden = nn.Sequential(
                nn.Linear(256 * 8 * 8, hidden_dim),
                nn.ReLU()
            )
    
    def _build_state_encoder(self, state_dim):
        """根据状态维度动态构建编码器"""
        if state_dim <= 50:
            # 小维度状态（如42维）：简单网络
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim),
                nn.ReLU()
            )
        elif state_dim <= 100:
            # 中等维度状态（如75维）：中等复杂度网络
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            )
        else:
            # 高维度状态（>100维）：更深的网络
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            )
    
    def update_state_dim(self, new_state_dim):
        """动态更新状态维度支持"""
        if new_state_dim != self.state_input_dim:
            self.state_input_dim = new_state_dim
            self._build_state_encoder(new_state_dim)
            print(f"状态编码器已更新为支持 {new_state_dim} 维输入")
    
    def forward(self, inputs):
        """
        Args:
            inputs: 可以是以下任一类型：
                - (batch_size, 3, H, W) - 输入图像
                - (batch_size, state_dim) - 输入状态向量
        Returns:
            embeddings: (batch_size, seq_len, output_dim) - 状态嵌入
        """
        batch_size = inputs.shape[0]
        
        # 判断输入类型：如果是4维且有图像尺寸，认为是图像；如果是2维，认为是状态向量
        if len(inputs.shape) == 4:  # 图像输入 (B, C, H, W)
            cnn_features = self.cnn_backbone(inputs)  # (B, 256, 8, 8)
            cnn_features = cnn_features.view(batch_size, -1)  # (B, 256*8*8)
            
            if self.state_input_dim is not None:
                # 同时支持图像和状态向量时，需要先映射到hidden_dim
                features = self.image_to_hidden(cnn_features)  # (B, hidden_dim)
            else:
                # 仅支持图像时，直接使用CNN特征
                features = cnn_features  # (B, 256*8*8)
                
        elif len(inputs.shape) == 2:  # 状态向量输入 (B, state_dim)
            if self.state_encoder is None:
                raise ValueError("状态向量输入需要在初始化时指定 state_input_dim")
            features = self.state_encoder(inputs)  # (B, hidden_dim)
        else:
            raise ValueError(f"不支持的输入维度: {inputs.shape}")
        
        embeddings = self.fc(features)  # (B, output_dim)
        # 扩展为序列维度以匹配transformer输入，原本用于文本，有特定的期望
        embeddings = embeddings.unsqueeze(1).repeat(1, 77, 1)  # (B, 77, output_dim)
        return embeddings

# 向后兼容的别名
RobotVisionEncoder = RobotStateEncoder

class ActionDecoder(nn.Module):
    """将latent解码为机器人动作 - 支持多种动作维度"""
    def __init__(self, latent_dim=16, action_dim=8, action_type="joint"):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_type = action_type  # "joint", "end_effector", "hybrid"
        
        self._build_decoder()
    
    def _build_decoder(self):
        """根据动作维度和类型构建解码器"""
        if self.action_dim == 4:
            # 4维末端位置控制 (x, y, z, gripper)
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim),
                nn.Tanh()  # 末端位置通常需要归一化
            )
        elif self.action_dim == 6:
            # 6维末端位置+姿态控制 (x, y, z, roll, pitch, yaw)
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim)
            )
        elif self.action_dim == 7:
            # 7维关节控制
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, self.action_dim)
            )
        elif self.action_dim == 8:
            # 8维关节控制（包括夹爪）
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_dim)
            )
        else:
            # 通用动作维度
            hidden_dim = max(64, self.action_dim * 4)
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.action_dim)
            )
    
    def forward(self, latents):
        """
        Args:
            latents: (batch_size, latent_dim, H, W) - 扩散模型输出
        Returns:
            actions: (batch_size, action_dim) - 机器人动作
        """
        # 全局平均池化
        pooled = latents.mean(dim=[2, 3])  # (B, latent_dim)
        actions = self.decoder(pooled)  # (B, action_dim)
        return actions

@torch.no_grad()
def robot_pipeline_with_logprob(
    self,
    state_inputs: Union[torch.Tensor, List[Image.Image], List[torch.Tensor]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "tensor",
    return_dict: bool = True,
    determistic: bool = False,
    state_encoder: Optional[RobotStateEncoder] = None,
    action_decoder: Optional[ActionDecoder] = None,
):
    """
    机器人操作管线：从输入状态（图像或状态向量）生成下一步动作
    
    Args:
        state_inputs: 当前状态输入，可以是：
            - 图像: tensor (B, 3, H, W) 或 PIL图像列表
            - 状态向量: tensor (B, state_dim) 或 状态向量列表
        state_encoder: 状态编码器（替代原vision_encoder）
        action_decoder: 动作解码器
        其他参数与原SD3管线类似
    
    Returns:
        actions: 预测的机器人动作
        all_latents: 所有中间latent
        all_log_probs: 所有步骤的对数概率
    """
    
    if state_encoder is None:
        raise ValueError("state_encoder is required for robot pipeline")
    if action_decoder is None:
        raise ValueError("action_decoder is required for robot pipeline")
    
    height = height or 64  # 动作空间较小，使用更小的latent尺寸
    width = width or 64
    
    # 1. 处理输入状态（图像或状态向量）
    if isinstance(state_inputs, list):
        if len(state_inputs) > 0 and isinstance(state_inputs[0], Image.Image):
            # PIL图像列表 -> tensor
            state_inputs = torch.stack([
                torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
                for img in state_inputs
            ])
        elif len(state_inputs) > 0 and isinstance(state_inputs[0], torch.Tensor):
            # 状态向量列表 -> tensor
            state_inputs = torch.stack(state_inputs)
        else:
            raise ValueError("不支持的状态输入列表类型")
    
    batch_size = state_inputs.shape[0]
    device = self._execution_device
    state_inputs = state_inputs.to(device)
    
    # 2. 编码状态输入为条件嵌入
    state_embeds = state_encoder(state_inputs)  # (B, seq_len, embed_dim)
    pooled_state_embeds = state_embeds.mean(dim=1)  # (B, embed_dim)
    
    # 3. 准备latent变量（动作空间）
    num_channels_latents = 16  # 可根据动作复杂度调整
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        state_embeds.dtype,
        device,
        generator,
        latents,
    )
    
    # 4. 准备时间步
    scheduler_kwargs = {}
    try:
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
    except:
        # 如果retrieve_timesteps失败，使用简单的线性时间步
        print("retrieve_timesteps failed, using simple linear time steps")
        timesteps = torch.linspace(1000, 0, num_inference_steps).to(device)
    
    all_latents = [latents]
    all_log_probs = []
    
    # 5. 扩散去噪循环
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
                
            # 扩展latents用于classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])
            
            # transformer前向传播
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=state_embeds,  # 使用状态嵌入而非文本嵌入
                pooled_projections=pooled_state_embeds,
                return_dict=False,
            )[0]
            
            # classifier-free guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
            # SDE步骤
            latents, log_prob, _, _ = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                determistic=determistic,
            )
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            
            progress_bar.update()
    
    # 6. 解码为动作
    actions = action_decoder(latents)  # (B, action_dim)
    
    if output_type == "latent":
        return latents, all_latents, all_log_probs
    
    if not return_dict:
        return (actions, all_latents, all_log_probs)
    
    return {
        "actions": actions,
        "latents": all_latents, 
        "log_probs": all_log_probs
    } 