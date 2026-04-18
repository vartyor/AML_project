import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EmbeddingExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()

        # --- GraphSAGE 레이어 3층 ---
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, embed_dim)

        # --- 정규화 ---
        self.dropout = nn.Dropout(p=0.3)

        # --- 이진 분류 헤드 (0: 정상, 1: 사기) ---
        self.classifier = nn.Linear(embed_dim, 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # 마지막 레이어는 활성화 없이 임베딩만 반환
        x = self.conv3(x, edge_index)
        return x

    # ------------------------------------------------------------------
    def classify(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.forward(x, edge_index)
        return self.classifier(emb)
