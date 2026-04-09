import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- 1. GNN 모델 클래스 정의 ---
class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, embed_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, embed_dim)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(embed_dim, 2)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index) 
        return x

@st.cache_resource
def load_resources():
    graph_dict = torch.load('model/processed_graph_data.pt', map_location='cpu')
    in_dim = graph_dict['x'].shape[1]
    extractor = EmbeddingExtractor(in_channels=in_dim, hidden_channels=128, embed_dim=64)
    extractor.load_state_dict(torch.load('gnn_model.pth', map_location='cpu'))
    extractor.eval()
    
    xgb_data = joblib.load('fraud_model.pkl')
    return graph_dict, extractor, xgb_data['xgb_model'], xgb_data['all_feature_names']

graph_dict, extractor, xgb_model, all_feature_names = load_resources()

# --- [수정 포인트 1] 사기 의심 계좌 우선 정렬 로직 ---
@st.cache_data
def get_sorted_test_nodes(_graph_dict, _extractor, _xgb_model, _all_feature_names):
    test_mask = _graph_dict['test_mask']
    test_idx = np.where(test_mask.numpy() == True)[0]
    
    with torch.no_grad():
        # 1. 모든 노드의 GNN 임베딩 추출 (111,427개 전체)
        all_embeddings = _extractor(_graph_dict['x'], _graph_dict['edge_index'])
        
        # 2. 테스트 노드에 해당하는 임베딩만 슬라이싱 (22,286, 64)
        test_embeddings = all_embeddings[test_idx].numpy()
        
        # [수정된 부분] 테스트 노드들에 대한 원본 피처(마지막 10개 열) 추출
        # 행은 test_idx 전체를 선택하고, 열은 마지막 10개를 선택해야 합니다.
        test_orig = _graph_dict['x'][test_idx, -10:].numpy() 
        
        # 3. 임베딩(64열) + 원본피처(10열) 결합 -> (22286, 74)
        X_test = np.hstack([test_embeddings, test_orig])
        
        # 4. XGBoost로 사기 확률 예측
        probs = _xgb_model.predict_proba(X_test)[:, 1]
        
    # 5. 결과 정리 및 정렬
    risk_df = pd.DataFrame({
        'node_idx': test_idx,
        'fraud_prob': probs,
        'key_feature': _graph_dict['x'][test_idx, -10].numpy() # 첫 번째 원본 피처 값
    })
    
    risk_df = risk_df.sort_values(by='fraud_prob', ascending=False)
    return risk_df

# 정렬된 위험 계좌 데이터프레임 가져오기
risk_df = get_sorted_test_nodes(graph_dict, extractor, xgb_model, all_feature_names)
high_risk_indices = risk_df['node_idx'].tolist()

# --- UI 레이아웃 ---
st.set_page_config(page_title="자금 세탁 경로 탐지 시스템", layout="wide")
st.title("🛡️ 자금 세탁 거점 분석 대시보드")

# 2. 사이드바에서 테스트 노드 선택
with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.info("💡 아래 리스트는 모델이 판단한 **사기 의심 확률이 높은 순**으로 정렬되어 있습니다.")
    
    # [수정 포인트 2] 리스트 출력 방식 개선
    def format_node_label(idx):
        row = risk_df[risk_df['node_idx'] == idx].iloc[0]
        prob = row['fraud_prob']
        feat = row['key_feature']
        return f"ID: {idx} (위험도: {prob:.1%}) | 값: {feat:.4f}"

    selected_idx = st.selectbox(
        "분석할 거래(노드) 선택", 
        high_risk_indices,
        format_func=format_node_label
    )
    analyze_btn = st.button("🔍 상세 분석 실행")

# 3. 메인 분석 영역
if analyze_btn:
    with torch.no_grad():
        # 임베딩 추출
        all_embeddings = extractor(graph_dict['x'], graph_dict['edge_index'])
        node_emb = all_embeddings[selected_idx].reshape(1, -1).numpy()
        
        # 원본 피처 (뒤에서 10개)
        node_orig = graph_dict['x'][selected_idx][-10:].reshape(1, -1).numpy()
        
        # 최종 입력 (74개 피처)
        X_input = np.hstack([node_emb, node_orig])
        X_input_df = pd.DataFrame(X_input, columns=all_feature_names)

    # 예측 확률 (미리 계산된 값 사용 가능하지만 정확성을 위해 재계산)
    prob = xgb_model.predict_proba(X_input)[0][1]
    
    # 상단 메트릭 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("선택된 노드", selected_idx)
    c2.metric("자금 세탁 통로 의심 점수", f"{prob:.2%}")
    
    if prob > 0.5:
        c3.error("🚨 자금 세탁 통로 의심 계좌")
    else:
        c3.success("✅ 정상 거래 계좌")

    st.divider()
    
    # 결과 분석 (SHAP)
    st.write("### 📊 왜 이렇게 판단했나요? (판단 근거)")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_input_df)

    col1, col2 = st.columns([1.5, 1])
    
    top_feature = all_feature_names[np.argmax(np.abs(shap_values[0]))]

    st.info(f"#### 💡 분석 핵심 요약: 가장 큰 영향을 준 요인은 **'{top_feature}'**입니다.")


    with col1:
        st.write("#### 1. 피처별 기여도 (영향력 TOP 10)")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[0], 
            feature_names=all_feature_names,
            max_display=10,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.write("#### 2. 원본 피처 데이터 정보")
        orig_feature_names = all_feature_names[-10:]
        orig_df = pd.DataFrame(node_orig, columns=orig_feature_names)
        st.table(orig_df.T.rename(columns={0: "값"}))

    st.success(f"""
    **💡 분석 결과 요약:**
    - 현재 이 거래는 **{prob:.2%}**의 확률로 사기일 가능성이 있습니다.
    - 위 그래프에서 **오른쪽(분홍색/양수)** 막대는 사기 확률을 **증가**시키는 요인입니다.
    - **왼쪽(하늘색/음수)** 막대는 사기 확률을 **감소**시키는 요인입니다.
    - `GNN_Emb`로 시작하는 항목은 인접한 노드(거래처)와의 관계에서 도출된 특징입니다.
    """)
    
else:
    st.write("👈 왼쪽 사이드바에서 노드를 선택하고 분석 버튼을 눌러주세요. (리스트 상단에 위험 계좌가 배치되어 있습니다)")
