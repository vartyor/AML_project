"""
models/embedding_extractor.py
-------------------------------
GraphSAGE 기반 노드 임베딩 추출 모델.
SAGEConv 3층 + Dropout + Linear 분류기로 구성됩니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EmbeddingExtractor(nn.Module):
    """
    GraphSAGE 기반 노드 임베딩 추출기.

    그래프 내 각 노드의 이웃 집계(neighbor aggregation)를 통해
    64차원 임베딩을 생성하고, 이진 분류(사기 / 정상)를 수행합니다.

    Args:
        in_channels  (int): 입력 노드 피처 차원수
        hidden_channels (int): SAGEConv 중간 레이어 차원수 (default: 128)
        embed_dim    (int): 최종 임베딩 차원수 (default: 64)
    """

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
        """
        순전파 — 임베딩만 반환합니다.
        분류 레이어는 XGBoost가 대체하므로 호출하지 않습니다.

        Args:
            x          (Tensor): 노드 피처 행렬  [num_nodes, in_channels]
            edge_index (Tensor): 엣지 인덱스     [2, num_edges]

        Returns:
            Tensor: 노드 임베딩 행렬 [num_nodes, embed_dim]
        """
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
        """
        임베딩을 거쳐 분류 로짓을 반환합니다.
        독립 학습 또는 end-to-end 평가 시 사용합니다.

        Returns:
            Tensor: 분류 로짓 [num_nodes, 2]
        """
        emb = self.forward(x, edge_index)
        return self.classifier(emb)
