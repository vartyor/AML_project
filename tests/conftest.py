"""
pytest 전역 설정 및 공유 픽스처
"""
import pytest
import os
import sys

# 프로젝트 루트를 sys.path에 추가 (GitHub Actions 환경 대응)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """커스텀 마커 등록"""
    config.addinivalue_line(
        "markers", "slow: 실행 시간이 긴 테스트 (CI에서 -m 'not slow'로 제외 가능)"
    )
    config.addinivalue_line(
        "markers", "integration: 외부 서비스 연동 테스트 (Neo4j, API 등)"
    )
    config.addinivalue_line(
        "markers", "requires_model: 실제 모델 파일(.pth, .pkl)이 필요한 테스트"
    )


@pytest.fixture(scope="session")
def project_root():
    """프로젝트 루트 경로"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def model_files_exist(project_root):
    """모델 파일 존재 여부 확인"""
    gnn_exists = os.path.exists(os.path.join(project_root, "gnn_model.pth"))
    xgb_exists = os.path.exists(os.path.join(project_root, "fraud_model.pkl"))
    graph_exists = os.path.exists(os.path.join(project_root, "model", "processed_graph_data.pt"))
    return {
        "gnn": gnn_exists,
        "xgb": xgb_exists,
        "graph": graph_exists,
        "all": gnn_exists and xgb_exists and graph_exists,
    }
