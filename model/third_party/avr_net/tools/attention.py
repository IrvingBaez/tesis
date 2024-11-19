import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, *x):
		if len(x) > 1:
			return x
		else:
			return x[0]


'''
Self attention recieves a tensor of shape ([N, F, 512, 7, 7])
and must return a tensor of shape ([N, 1, 512, 7, 7])
'''
class PickFirst(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x = x[:, 0:1, ...]

		return x


import torch
import torch.nn as nn

class SelfAttentionClassToken(nn.Module):
	def __init__(self):
		super(SelfAttentionClassToken, self).__init__()
		input_dim = 512 * 7 * 7  # Input embedding dimension
		self.input_dim = input_dim
		self.model_dim = 512  # Transformer embedding dimension
		self.projection = nn.Linear(input_dim, self.model_dim)

		# Initialize the class token
		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
		nn.init.normal_(self.cls_token, std=0.02)

		# Transformer encoder layer
		encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=8)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

		# Output projection to reshape back to original dimensions
		self.output_projection = nn.Linear(self.model_dim, input_dim)


	def forward(self, x):
		B, N, C, H, W = x.shape  # Input shape: [B, N, 512, 7, 7]
		D = self.input_dim  # D = 512 * 7 * 7

		# Flatten spatial dimensions and project to embedding dimension
		x = x.view(B, N, D)            # Shape: [B, N, 512 * 7 * 7]
		x = self.projection(x)         # Shape: [B, N, d_model]

		# Prepare class token
		cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, d_model]

		# Concatenate class token with the sequence
		x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, N+1, d_model]

		# Transformer expects input as [seq_len, batch_size, embedding_dim]
		x = x.transpose(0, 1)  # Shape: [N+1, B, d_model]

		# Pass through the Transformer encoder
		x = self.transformer_encoder(x)  # Shape: [N+1, B, d_model]

		# Revert back to [batch_size, seq_len, embedding_dim]
		x = x.transpose(0, 1)  # Shape: [B, N+1, d_model]

		# Extract the class token output
		cls_output = x[:, 0, :]  # Shape: [B, d_model]

		# Project back to original dimension and reshape
		cls_output = self.output_projection(cls_output)  # Shape: [B, D]
		cls_output = cls_output.view(B, 1, C, H, W)      # Shape: [B, 1, 512, 7, 7]

		return cls_output


'''
Cross attention recieves two tensors:
	Video shape: ([N, 1, 512, 7, 7])
	Audio shape: ([N, 1, 256, 7, 7])

And must return a tensor of shape:
	([N, 1, 768, 7, 7])
'''
class Concat(nn.Module):
	def __init__(self):
		super().__init__()


	def forward(self, feature_a, feature_b):
		feats = torch.cat((feature_a, feature_b), dim=2)

		return feats


class FusionCrossAttention(nn.Module):
	def __init__(self, dim_a, dim_b):
		super().__init__()

		self.key_a 		= nn.Linear(dim_a, dim_a)
		self.query_a 	= nn.Linear(dim_a, dim_a)
		self.value_a 	= nn.Linear(dim_a, dim_a)

		self.key_b 		= nn.Linear(dim_b, dim_b)
		self.query_b 	= nn.Linear(dim_b, dim_b)
		self.value_b 	= nn.Linear(dim_b, dim_b)


	def forward(self, feature_a, feature_b):
		N, _, C_a, H, W = feature_a.shape
		_, _, C_b, _, _ = feature_b.shape

		feature_a = feature_a.view(N, C_a, H * W) # [N, 512, 49]
		feature_b = feature_b.view(N, C_b, H * W) # [N, 256, 49]

		KA = self.key_a(feature_a)
		QA = self.query_a(feature_a)
		VA = self.value_a(feature_a)

		KB = self.key_b(feature_b)
		QB = self.query_b(feature_b)
		VB = self.value_b(feature_b)

		dA = KA.shape[-1]
		dB = KB.shape[-1]

		# Cross Attention: softmax( QA * KᵀA / sqrt(dA) ) * VA
		# RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [512, 256] but got: [512, 512].
		attn_a = torch.matmul(QB, KA.transpose(-2, -1)) / torch.sqrt(torch.tensor(dA, dtype=torch.float32))
		attn_a = F.softmax(attn_a, dim=-1)
		fused_a = torch.matmul(attn_a, VA)

		# Cross Attention: softmax( QAKBᵀ / sqrt(dB) ) * VB
		attn_b = torch.matmul(QA, KB.transpose(-2, -1)) / torch.sqrt(torch.tensor(dB, dtype=torch.float32))
		attn_b = F.softmax(attn_b, dim=-1)
		fused_b = torch.matmul(attn_b, VB)

		fAB = torch.cat([fused_a, fused_b], dim=1)

		fAB = fAB.view(N, -1, H, W)  # Reshape to [N, 768, 7, 7]
		fAB = fAB.unsqueeze(1)  # Add singleton dimension for [N, 1, 768, 7, 7]

		return fAB
