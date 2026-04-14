"""
loaders/resource_loader.py
---------------------------
모델 / 그래프 데이터 / 임베딩 / 위험 순위를 로드·캐싱하는 모듈.

Streamlit @st.cache_resource / @st.cache_data 데코레이터를 사용해
앱 재실행 시 중복 연산을 방지합니다.
"""

import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st

from models.embedding_extractor import EmbeddingExtractor


# ======================================================================
# 1. 모델 및 데이터 로드
# ======================================================================

@st.cache_resource
def load_resources(
    graph_path: str = "model/processed_graph_data.pt",
    gnn_path: str = "gnn_model.pth",
    xgb_path: str = "fraud_model.pkl",
) -> tuple:
    """
    GNN 모델, XGBoost 모델, 그래프 데이터를 로드합니다.
    @st.cache_resource 로 한 번만 실행됩니다.

    Args:
        graph_path (str): 전처리된 그래프 데이터 경로 (.pt)
        gnn_path   (str): GNN 가중치 파일 경로 (.pth)
        xgb_path   (str): XGBoost 모델 파일 경로 (.pkl)

    Returns:
        tuple: (graph_dict, extractor, xgb_model, all_feature_names)
    """
    # 1-1. 그래프 데이터 로드
    graph_dict = torch.load(graph_path, map_location="cpu")

    # 1-2. GNN 모델 초기화 및 가중치 로드
    in_dim = graph_dict["x"].shape[1]
    extractor = EmbeddingExtractor(
        in_channels=in_dim,
        hidden_channels=128,
        embed_dim=64,
    )
    extractor.load_state_dict(torch.load(gnn_path, map_location="cpu"))
    extractor.eval()

    # 1-3. XGBoost 모델 및 피처명 로드
    xgb_data = joblib.load(xgb_path)
    xgb_model = xgb_data["xgb_model"]
    all_feature_names = xgb_data["all_feature_names"]

    return graph_dict, extractor, xgb_model, all_feature_names


# ======================================================================
# 2. 전체 노드 임베딩 사전 계산
# ======================================================================

@st.cache_data
def get_all_embeddings_cached(
    _extractor: EmbeddingExtractor,
    _x: torch.Tensor,
    _edge_index: torch.Tensor,
) -> np.ndarray:
    """
    전체 그래프에 대해 GNN 임베딩을 한 번만 계산하고 numpy로 반환합니다.

    Args:
        _extractor  : EmbeddingExtractor 인스턴스 (언더스코어 prefix → 캐시 키 제외)
        _x          : 노드 피처 텐서 [num_nodes, in_channels]
        _edge_index : 엣지 인덱스 텐서 [2, num_edges]

    Returns:
        np.ndarray: 전체 노드 임베딩 [num_nodes, embed_dim]
    """
    with torch.no_grad():
        embeddings = _extractor(_x, _edge_index)
    return embeddings.numpy()


# ======================================================================
# 3. 위험 순위 DataFrame 생성
# ======================================================================

@st.cache_data
def get_sorted_test_nodes(
    _all_embs: np.ndarray,
    _graph_dict: dict,
    _xgb_model,
    _all_feature_names: list,
) -> pd.DataFrame:
    """
    테스트 노드 전체에 대해 사기 확률을 예측하고 내림차순 정렬된
    DataFrame을 반환합니다.

    Pipeline:
        test_mask 인덱스 추출
        → GNN 임베딩(64) + 원본 피처(10) 결합(74차원)
        → XGBoost predict_proba
        → 사기 확률 내림차순 정렬

    Returns:
        pd.DataFrame: columns = ['node_idx', 'fraud_prob', 'key_feature']
    """
    test_mask = _graph_dict["test_mask"]
    test_idx = np.where(test_mask.numpy())[0]

    # 임베딩 (미리 계산된 값에서 인덱싱)
    test_embeddings = _all_embs[test_idx]

    # 원본 피처 (뒤에서 10개)
    test_orig = _graph_dict["x"][test_idx, -10:].numpy()

    # XGBoost 입력 구성
    X_test = np.hstack([test_embeddings, test_orig])

    # 사기 확률 예측
    probs = _xgb_model.predict_proba(X_test)[:, 1]

    risk_df = pd.DataFrame(
        {
            "node_idx": test_idx,
            "fraud_prob": probs,
            "key_feature": _graph_dict["x"][test_idx, -10].numpy(),
        }
    ).sort_values(by="fraud_prob", ascending=False)

    return risk_df
