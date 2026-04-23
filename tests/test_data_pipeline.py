"""
테스트: 데이터 파이프라인 및 전처리
- 데이터셋 로딩, 피처 엔지니어링, 그래프 구성 관련 테스트
"""
import pytest
import numpy as np
import pandas as pd
import torch
import os


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_transaction_df():
    """PaySim 형식의 더미 트랜잭션 데이터"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "step": np.random.randint(1, 744, n),
        "type": np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n),
        "amount": np.random.exponential(10000, n),
        "nameOrig": [f"C{i:010d}" for i in range(n)],
        "oldbalanceOrg": np.random.uniform(0, 100000, n),
        "newbalanceOrig": np.random.uniform(0, 100000, n),
        "nameDest": [f"C{i+n:010d}" for i in range(n)],
        "oldbalanceDest": np.random.uniform(0, 100000, n),
        "newbalanceDest": np.random.uniform(0, 100000, n),
        "isFraud": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "isFlaggedFraud": np.zeros(n, dtype=int),
    })


@pytest.fixture
def dummy_graph_dict():
    """더미 그래프 딕셔너리 (processed_graph_data.pt 형식 모방)"""
    num_nodes = 50
    in_channels = 10
    num_edges = 80

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))

    # 마스크 생성
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:30]] = True
    test_mask[perm[30:]] = True

    # 피처 이름 (GNN 임베딩 + 수치 피처)
    feature_names = [f"embed_{i}" for i in range(64)] + [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]

    return {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "feature_names": feature_names,
    }


# ──────────────────────────────────────────────
# 1. 트랜잭션 데이터 기본 검증
# ──────────────────────────────────────────────

class TestTransactionData:

    def test_required_columns(self, sample_transaction_df):
        """필수 컬럼 존재 여부 확인"""
        required = ["step", "type", "amount", "nameOrig", "nameDest",
                    "oldbalanceOrg", "newbalanceOrig", "isFraud"]
        for col in required:
            assert col in sample_transaction_df.columns, f"컬럼 누락: {col}"

    def test_no_negative_amounts(self, sample_transaction_df):
        """거래 금액이 음수가 아닌지 확인"""
        assert (sample_transaction_df["amount"] >= 0).all()

    def test_fraud_label_binary(self, sample_transaction_df):
        """isFraud가 0 또는 1만 포함하는지 확인"""
        assert set(sample_transaction_df["isFraud"].unique()).issubset({0, 1})

    def test_transaction_type_valid(self, sample_transaction_df):
        """거래 유형이 유효한 값인지 확인"""
        valid_types = {"TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"}
        actual_types = set(sample_transaction_df["type"].unique())
        assert actual_types.issubset(valid_types), f"유효하지 않은 거래 유형: {actual_types - valid_types}"

    def test_dataframe_not_empty(self, sample_transaction_df):
        """DataFrame이 비어있지 않은지 확인"""
        assert len(sample_transaction_df) > 0


# ──────────────────────────────────────────────
# 2. 그래프 데이터 구조 검증
# ──────────────────────────────────────────────

class TestGraphData:

    def test_graph_dict_keys(self, dummy_graph_dict):
        """그래프 딕셔너리에 필수 키가 있는지 확인"""
        required_keys = ["x", "edge_index", "y", "train_mask", "test_mask"]
        for key in required_keys:
            assert key in dummy_graph_dict, f"키 누락: {key}"

    def test_node_feature_shape(self, dummy_graph_dict):
        """노드 피처 텐서 shape 확인"""
        x = dummy_graph_dict["x"]
        assert x.dim() == 2, "노드 피처는 2D 텐서여야 합니다."
        assert x.shape[0] > 0, "노드가 1개 이상이어야 합니다."

    def test_edge_index_shape(self, dummy_graph_dict):
        """엣지 인덱스가 (2, num_edges) shape인지 확인"""
        edge_index = dummy_graph_dict["edge_index"]
        assert edge_index.shape[0] == 2, "edge_index의 첫 번째 차원은 2여야 합니다."

    def test_edge_index_within_bounds(self, dummy_graph_dict):
        """엣지 인덱스가 노드 수 범위 내인지 확인"""
        num_nodes = dummy_graph_dict["x"].shape[0]
        edge_index = dummy_graph_dict["edge_index"]
        assert edge_index.max() < num_nodes, "엣지 인덱스가 노드 수를 초과합니다."
        assert edge_index.min() >= 0, "엣지 인덱스에 음수가 있습니다."

    def test_masks_no_overlap(self, dummy_graph_dict):
        """train/test 마스크가 겹치지 않는지 확인"""
        train_mask = dummy_graph_dict["train_mask"]
        test_mask = dummy_graph_dict["test_mask"]
        overlap = (train_mask & test_mask).sum()
        assert overlap == 0, f"train/test 마스크가 {overlap}개 노드에서 겹칩니다."

    def test_label_tensor_dtype(self, dummy_graph_dict):
        """레이블 텐서가 정수형인지 확인"""
        y = dummy_graph_dict["y"]
        assert y.dtype in (torch.long, torch.int64, torch.int32), \
            f"레이블은 정수형이어야 합니다. 실제: {y.dtype}"

    def test_label_binary(self, dummy_graph_dict):
        """레이블이 0 또는 1만 포함하는지 확인"""
        y = dummy_graph_dict["y"]
        unique_labels = y.unique().tolist()
        assert all(l in [0, 1] for l in unique_labels), \
            f"이진 레이블이 아닙니다: {unique_labels}"


# ──────────────────────────────────────────────
# 3. 피처 엔지니어링 검증
# ──────────────────────────────────────────────

class TestFeatureEngineering:

    def test_balance_diff_feature(self, sample_transaction_df):
        """잔액 차이 피처 계산 검증"""
        df = sample_transaction_df.copy()
        df["balance_diff"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
        assert "balance_diff" in df.columns
        assert df["balance_diff"].notna().all()

    def test_amount_log_transform(self, sample_transaction_df):
        """금액 로그 변환 정상 동작 확인"""
        df = sample_transaction_df.copy()
        df["log_amount"] = np.log1p(df["amount"])
        assert (df["log_amount"] >= 0).all()
        assert df["log_amount"].notna().all()

    def test_type_encoding(self, sample_transaction_df):
        """거래 유형 원핫 인코딩 확인"""
        encoded = pd.get_dummies(sample_transaction_df["type"], prefix="type")
        assert encoded.shape[0] == len(sample_transaction_df)
        assert encoded.shape[1] >= 1
