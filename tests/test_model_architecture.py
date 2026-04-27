"""
테스트: GNN 모델 아키텍처 및 순전파
- 모델 파일(.pth, .pkl) 없이도 실행 가능한 구조 검증 테스트
"""
import pytest
import torch
import numpy as np


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def dummy_graph():
    """간단한 더미 그래프 데이터 생성 (5 노드, 6 엣지)"""
    num_nodes = 5
    in_channels = 10
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 2],
    ], dtype=torch.long)
    return x, edge_index, in_channels


@pytest.fixture
def embedding_extractor(dummy_graph):
    """EmbeddingExtractor 인스턴스 생성"""
    from models.embedding_extractor import EmbeddingExtractor
    _, _, in_channels = dummy_graph
    model = EmbeddingExtractor(
        in_channels=in_channels,
        hidden_channels=32,
        embed_dim=16,
    )
    model.eval()
    return model


# ──────────────────────────────────────────────
# 1. EmbeddingExtractor 구조 테스트
# ──────────────────────────────────────────────

class TestEmbeddingExtractor:

    def test_model_instantiation(self, embedding_extractor):
        """모델이 정상 생성되는지 확인"""
        from models.embedding_extractor import EmbeddingExtractor
        assert isinstance(embedding_extractor, EmbeddingExtractor)

    def test_forward_output_shape(self, embedding_extractor, dummy_graph):
        """forward() 출력 shape이 (num_nodes, embed_dim)인지 확인"""
        x, edge_index, _ = dummy_graph
        with torch.no_grad():
            out = embedding_extractor(x, edge_index)
        assert out.shape == (5, 16), f"예상 (5, 16), 실제 {out.shape}"

    def test_classify_output_shape(self, embedding_extractor, dummy_graph):
        """classify() 출력 shape이 (num_nodes, 2)인지 확인 (이진 분류)"""
        x, edge_index, _ = dummy_graph
        with torch.no_grad():
            logits = embedding_extractor.classify(x, edge_index)
        assert logits.shape == (5, 2), f"예상 (5, 2), 실제 {logits.shape}"

    def test_embedding_is_float(self, embedding_extractor, dummy_graph):
        """임베딩이 float 타입인지 확인"""
        x, edge_index, _ = dummy_graph
        with torch.no_grad():
            out = embedding_extractor(x, edge_index)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, embedding_extractor, dummy_graph):
        """출력에 NaN 없는지 확인"""
        x, edge_index, _ = dummy_graph
        with torch.no_grad():
            out = embedding_extractor(x, edge_index)
        assert not torch.isnan(out).any(), "임베딩에 NaN 값이 존재합니다."

    def test_layer_count(self, embedding_extractor):
        """SAGEConv 레이어가 3개인지 확인"""
        from torch_geometric.nn import SAGEConv
        sage_layers = [m for m in embedding_extractor.modules() if isinstance(m, SAGEConv)]
        assert len(sage_layers) == 3, f"SAGEConv 레이어 수: {len(sage_layers)}"

    def test_eval_mode(self, embedding_extractor):
        """모델이 eval 모드인지 확인"""
        assert not embedding_extractor.training

    def test_parameter_count(self, embedding_extractor):
        """파라미터가 존재하는지 확인"""
        total_params = sum(p.numel() for p in embedding_extractor.parameters())
        assert total_params > 0, "파라미터가 없습니다."


# ──────────────────────────────────────────────
# 2. 임베딩 추출 유틸 테스트
# ──────────────────────────────────────────────

class TestEmbeddingExtraction:

    def test_numpy_conversion(self, embedding_extractor, dummy_graph):
        """임베딩을 numpy 배열로 변환 가능한지 확인"""
        x, edge_index, _ = dummy_graph
        with torch.no_grad():
            out = embedding_extractor(x, edge_index)
        # detach().cpu() 를 먼저 호출해야 CPU-only 빌드에서도 안전하게 변환 가능
        arr = out.detach().cpu().numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 16)

    def test_deterministic_output(self, embedding_extractor, dummy_graph):
        """eval 모드에서 동일 입력에 동일 출력인지 확인"""
        x, edge_index, _ = dummy_graph
        with torch.no_grad():
            out1 = embedding_extractor(x, edge_index)
            out2 = embedding_extractor(x, edge_index)
        assert torch.allclose(out1, out2), "동일 입력에 다른 결과가 나왔습니다."
