import torch
import torch.nn as nn
import timm

from models.vit_small import ViT
from models import *

# class Router_Dense(nn.Module):
#     def __init__(self, num_experts, image_size = 32):
#         super(Router_Dense, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         cnn_output_size = (image_size // 8) * (image_size // 8) * 64
#         self.fc = nn.Linear(cnn_output_size, num_experts)
#
#     def forward(self, x):
#         x = self.conv_layers(x)
#
#         x = x.view(x.size(0), -1)
#
#         gate_weights = self.fc(x)
#
#         gate_weights = torch.softmax(gate_weights, dim=1)
#         return gate_weights
#
#
# class Router_Top_1(nn.Module):
#     def __init__(self, num_experts, image_size = 32):
#         super(Router_Top_1, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         cnn_output_size = (image_size // 8) * (image_size // 8) * 64
#         self.fc = nn.Linear(cnn_output_size, num_experts)
#     def forward(self, x):
#         x = self.conv_layers(x)
#
#         x = x.view(x.size(0), -1)
#
#         gate_scores = self.fc(x)
#
#         top1_experts = torch.argmax(gate_scores, dim=1)
#         return top1_experts


class Router_Dense(nn.Module):
    def __init__(self, num_experts, size):
        super(Router_Dense, self).__init__()
        self.gate = nn.Linear(3*size*size, num_experts)
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        gate_weights = torch.softmax(self.gate(x_flat), dim=1)
        return gate_weights


class Router_Top_1(nn.Module):
    def __init__(self, num_experts, size):
        super(Router_Top_1, self).__init__()
        self.gate = nn.Linear(3*size*size, num_experts)
    def forward(self, x):
        #Top-1 strategy
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        gate_logits = self.gate(x_flat)
        top1_expert = torch.argmax(gate_logits, dim=1)
        return top1_expert


class ResnetExpert(nn.Module):
    def __init__(self):
        super(ResnetExpert, self).__init__()
        self.net = ResNet18()

    def forward(self, x):
        return self.net(x)


class ViTExpert(nn.Module):
    def __init__(self, size):
        super(ViTExpert, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, img_size=(size, size))
        # self.vit = timm.create_model('vit_small_patch16_224', img_size=(size, size))
        # self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, in_chans=3, img_size=(size, size))
        self.vit.head = nn.Linear(self.vit.head.in_features, 200)

    def forward(self, x):
        return self.vit(x)


class SmallViTExpert(nn.Module):
    def __init__(self, size, patch, dimhead):
        super(SmallViTExpert, self).__init__()
        self.vit = ViT(
        image_size=size,
        patch_size=patch,
        num_classes=10,
        dim=int(dimhead),
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )

    def forward(self, x):
        return self.vit(x)


class MOE_Resnet18_Top_1(nn.Module):
    def __init__(self, num_experts, num_classes, size):
        super(MOE_Resnet18_Top_1, self).__init__()
        self.experts = nn.ModuleList([ResnetExpert() for _ in range(num_experts)])
        self.router = Router_Top_1(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router
        batch_size = x.size(0)

        #Top-1 Strategy
        top1_expert_indices = self.router(x)

        # Forward the input through the selected expert
        outputs = torch.zeros(batch_size, 10).to(x.device)  # 10: dataset class
        for i in range(batch_size):
            expert_output = self.experts[top1_expert_indices[i]](x[i].unsqueeze(0)) # It should be done in batch-size
            outputs[i] = expert_output

        return outputs


class MOE_Resnet18_Dense(nn.Module):
    def __init__(self, num_experts, num_classes, size):
        super(MOE_Resnet18_Dense, self).__init__()
        self.experts = nn.ModuleList([ResnetExpert() for _ in range(num_experts)])
        self.router = Router_Dense(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router

        # Get gate weights
        gate_weights = self.router(x)

        # Compute expert outputs
        expert_outputs = [expert(x) for expert in self.experts]

        # Weighted sum of expert outputs
        outputs = sum(gate_weights[:, i].unsqueeze(1) * expert_outputs[i] for i in range(len(self.experts)))
        return outputs


class MOE_ViT(nn.Module):
    def __init__(self, num_experts, size):
        super(MOE_ViT, self).__init__()
        self.experts = nn.ModuleList([ViTExpert(size=size) for _ in range(num_experts)])
        self.router = Router_Dense(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router

        # Get gate weights
        gate_weights = self.router(x)

        # Compute expert outputs
        expert_outputs = [expert(x) for expert in self.experts]

        # Weighted sum of expert outputs
        outputs = sum(gate_weights[:, i].unsqueeze(1) * expert_outputs[i] for i in range(len(self.experts)))
        return outputs


class Dual_Model_resnet18(nn.Module):
    def __init__(self, alpha, size):
        super(Dual_Model_resnet18, self).__init__()
        self.alpha = alpha
        self.Smoe = MOE_Resnet18_Dense(num_experts=4, num_classes=10, size=size)
        self.Rmoe = MOE_Resnet18_Dense(num_experts=4, num_classes=10, size=size)

    def forward(self, x):
        outputs = (1- self.alpha) * self.Smoe(x) + self.alpha * self.Rmoe(x)
        return outputs


class MOE_ViTsmall_Top_1(nn.Module):
    def __init__(self, num_experts, num_classes, size, patch, dimhead):
        super(MOE_ViTsmall_Top_1, self).__init__()
        self.experts = nn.ModuleList([SmallViTExpert(size, patch, dimhead) for _ in range(num_experts)])
        self.router = Router_Top_1(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router
        batch_size = x.size(0)

        #Top-1 Strategy
        top1_expert_indices = self.router(x)

        # Forward the input through the selected expert
        outputs = torch.zeros(batch_size, 10).to(x.device)  # 10: dataset class
        for i in range(batch_size):
            expert_output = self.experts[top1_expert_indices[i]](x[i].unsqueeze(0))
            outputs[i] = expert_output

        return outputs


class MOE_ViTsmall_Dense(nn.Module):
    def __init__(self, num_experts, num_classes, size, patch, dimhead):
        super(MOE_ViTsmall_Dense, self).__init__()
        self.experts = nn.ModuleList([SmallViTExpert(size, patch, dimhead) for _ in range(num_experts)])
        self.router = Router_Dense(num_experts, size)

    def forward(self, x):
        # Flatten the input to (batch_size, -1) for the router

        # Get gate weights
        gate_weights = self.router(x)

        # Compute expert outputs
        expert_outputs = [expert(x) for expert in self.experts]

        # Weighted sum of expert outputs
        outputs = sum(gate_weights[:, i].unsqueeze(1) * expert_outputs[i] for i in range(len(self.experts)))
        return outputs