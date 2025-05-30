import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import NamedTuple
from huggingface_hub import hf_hub_download,list_repo_files
from safetensors.torch import load_file

def get_optimal_file(repo_id,layer,width):
    directory_path = f"layer_{layer}/width_{width}"
    files_with_l0s = [
            (f, int(f.split("_")[-1].split("/")[0]))
            for f in list_repo_files(repo_id, repo_type="model", revision="main")
            if f.startswith(directory_path) and f.endswith("params.npz")
        ]

    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return optimal_file

BANDWIDTH = 0.1
def rectangle_pt(x):
    return ((x > -0.5) & (x < 0.5)).to(x)

class Step(torch.autograd.Function):
    @staticmethod
    def forward(x, threshold):
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, threshold = inputs
        del output
        ctx.save_for_backward(x, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        x_grad = 0.0 * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(1.0 / BANDWIDTH)
            * rectangle_pt((x - threshold) / BANDWIDTH)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad
      
class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(x, threshold):
        return x * (x > threshold).to(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, threshold = inputs
        del output
        ctx.save_for_backward(x, threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / BANDWIDTH)
            * rectangle_pt((x - threshold) / BANDWIDTH)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad

class ForwardOutput(NamedTuple):

    fvu: Tensor
    """Fraction of variance unexplained."""

    sparsity_loss: Tensor
    """AuxK loss, if applicable."""
          
# This is from the GemmaScope tutorial
# https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp#scrollTo=WYfvS97fAFzq
class JumpReLUSAE(nn.Module):
  def __init__(self, sae_width, activations_size): 
      super().__init__()
      threshold = 0.001
      self.W_enc = nn.Parameter(torch.empty((activations_size, sae_width)))
      self.b_enc = nn.Parameter(torch.empty((sae_width,)))
      self.W_dec = nn.Parameter(torch.empty((sae_width, activations_size)))
      self.b_dec = nn.Parameter(torch.empty((activations_size,)))
      self.log_threshold = nn.Parameter(
          torch.tensor(np.log(threshold))
      )

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    # mask = (pre_acts > self.threshold)
    mask = (pre_acts > torch.exp(self.log_threshold))
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec
    
  def freeze_decoder(self):
      self.W_dec.requires_grad = False
      self.b_dec.requires_grad = False
          
  def __call__(self, x):
      # x = x - self.b_dec

      pre_activations = x @ self.W_enc + self.b_enc
      threshold = torch.exp(self.log_threshold)
      feature_magnitudes = JumpReLU.apply(pre_activations, threshold)
      x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

      reconstruction_error = x - x_reconstructed
      reconstruction_loss = torch.sum(reconstruction_error**2)
      
      # Compute per-example sparsity loss
      threshold = torch.exp(self.log_threshold)
      l0 = torch.sum(Step.apply(pre_activations, threshold))

      total_variance = (x - x.mean(0)).pow(2).sum()
      fvu=reconstruction_loss/total_variance
      return ForwardOutput(
          fvu,
          l0,
      )
  
  @classmethod
  def from_pretrained(cls, model_name_or_path,position="",device="cuda",is_hf=False):

    if is_hf:
        path_to_params = hf_hub_download(
        repo_id=model_name_or_path,
        filename=f"{position}/sae.safetensors",
        force_download=False,
        token  = "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"
        )
    else:
       path_to_params = model_name_or_path
    params=load_file(path_to_params)
    
    model = cls(params['W_enc'].shape[1], params['W_enc'].shape[0])
    model.load_state_dict(params)
    if device == "cuda":
        model.cuda()
    return model

# class JumpReLUSAE(nn.Module):
#   def __init__(self, d_model, d_sae):
#     super().__init__()
#     self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
#     self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
#     self.threshold = nn.Parameter(torch.zeros(d_sae))
#     self.b_enc = nn.Parameter(torch.zeros(d_sae))
#     self.b_dec = nn.Parameter(torch.zeros(d_model))

#   def encode(self, input_acts):
#     pre_acts = input_acts @ self.W_enc + self.b_enc
#     mask = (pre_acts > self.threshold)
#     # mask = (pre_acts > torch.exp(self.log_threshold))
#     acts = mask * torch.nn.functional.relu(pre_acts)
#     return acts

#   def decode(self, acts):
#     return acts @ self.W_dec + self.b_dec

#   def forward(self, acts):
#     acts = self.encode(acts)
#     recon = self.decode(acts)
#     return recon
  
# #   def __init__(self, sae_width, activations_size): 
# #       super().__init__()
# #       threshold = 0.001
# #       self.W_enc = nn.Parameter(torch.empty((activations_size, sae_width)))
# #       self.b_enc = nn.Parameter(torch.empty((sae_width,)))
# #       self.W_dec = nn.Parameter(torch.empty((sae_width, activations_size)))
# #       self.b_dec = nn.Parameter(torch.empty((activations_size,)))
# #       self.log_threshold = nn.Parameter(
# #           torch.tensor(np.log(threshold))
# #       )
    
#   def freeze_decoder(self):
#       self.W_dec.requires_grad = False
#       self.b_dec.requires_grad = False
          
#   def __call__(self, x):
#       # x = x - self.b_dec

#       pre_activations = x @ self.W_enc + self.b_enc
#       threshold = torch.exp(self.log_threshold)
#       feature_magnitudes = JumpReLU.apply(pre_activations, threshold)
#       x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

#       reconstruction_error = x - x_reconstructed
#       reconstruction_loss = torch.sum(reconstruction_error**2)
      
#       # Compute per-example sparsity loss
#     #   threshold = torch.exp(self.log_threshold)
#       l0 = torch.sum(Step.apply(pre_activations, threshold))

#       total_variance = (x - x.mean(0)).pow(2).sum()
#       fvu=reconstruction_loss/total_variance
#       return ForwardOutput(
#           fvu,
#           l0,
#       )
  
#   @classmethod
#   def from_pretrained(cls, model_name_or_path,position="",device="cuda"):
#     path_to_params = hf_hub_download(
#     repo_id=model_name_or_path,
#     filename=f"{position}/params.npz",
#     force_download=False,
#     )
#     params = np.load(path_to_params)
#     pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    
#     # path_to_params = hf_hub_download(
#     # repo_id=model_name_or_path,
#     # filename=f"{position}/sae.safetensors",
#     # force_download=False,
#     # )
#     # params=load_file(path_to_params)
    
#     model = cls(params['W_enc'].shape[0], params['W_enc'].shape[1])
#     model.load_state_dict(pt_params)
#     # if device == "cuda":
#     model.to(device)
#     return model



class JumpReLUSAE_Base(nn.Module):
  def __init__(self, d_model, d_sae):
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))
    
  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    #print(pre_acts.shape)
    #print(torch.nonzero(mask).shape)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon
  
  @classmethod
  def from_pretrained(cls, model_name_or_path,position="",device="cuda"):
    path_to_params = hf_hub_download(
    repo_id=model_name_or_path,
    filename=f"{position}/params.npz",
    force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
    model = cls(params['W_enc'].shape[0], params['W_enc'].shape[1])
    model.load_state_dict(pt_params)
    model.device = device
    return model