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

# --- 1. GNN 모델 클래스 정의 (보내주신 코드와 100% 동일하게) ---
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

# --- UI 레이아웃 ---
st.set_page_config(page_title="자금 세탁 경로 탐지 시스템", layout="wide")
st.title("🛡️ 자금 세탁 거점 분석 대시보드")

# 1. 사이드바에서 테스트 노드 선택
test_idx = np.where(graph_dict['test_mask'].numpy() == True)[0]

with st.sidebar:
    st.header("⚙️ 분석 설정")
    
    # 인덱스 선택 시 원본 피처의 일부 값을 요약해서 보여줌
    # 예: 마지막 10개 원본 피처 중 첫 번째 값을 '거래량'이라고 가정할 경우
    def format_node_label(idx):
        # graph_dict['x'][idx]에서 원본 피처가 있는 위치의 값을 가져옵니다.
        feature_summary = graph_dict['x'][idx][-10].item() # 첫 번째 원본 피처 값
        return f"📍 거래 ID: {idx} (Key Feature: {feature_summary:.4f})"

    selected_idx = st.selectbox(
        "분석할 거래(노드) 선택", 
        test_idx,
        format_func=format_node_label
    )
    analyze_btn = st.button("🔍 상세 분석 실행")

# 2. 메인 분석 영역
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

    # 예측
    prob = xgb_model.predict_proba(X_input)[0][1]
    
    # 상단 메트릭 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("선택된 노드", selected_idx)
    c2.metric("자금 세탁 통로 의심 점수", f"{prob:.2%}")
    c3.subheader("🚨 자금 세탁 통로 의심 계좌" if prob > 0.5 else "✅ 정상 거래 계좌")

    st.divider()
    

    # 결과 분석 (SHAP)
    st.write("### 📊 왜 이렇게 판단했나요? (판단 근거)")
    
    # SHAP Explainer 설정
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_input_df)

    col1, col2 = st.columns([1.5, 1])
    
    # SHAP 결과에 따른 지능형 설명 생성
    top_feature = all_feature_names[np.argmax(np.abs(shap_values[0]))]

    st.info(f"""
    #### 💡 분석 핵심 요약
    가장 큰 영향을 준 요인은 **'{top_feature}'**입니다.
    """)

    if "GNN_Emb" in top_feature:
        st.write("👉 이 거래는 개별 정보보다 **주변 관계망(네트워크)상의 위험도**가 매우 높게 측정되었습니다. 연결된 계좌들이 자금 세탁과 연관되었을 가능성이 큽니다.")
    else:
        st.write("👉 이 거래는 **거래 자체의 수치(금액, 시간 등)**에서 강한 이상 징후가 발견되었습니다.")


    with col1:
        st.write("#### 1. 피처별 기여도 (영향력 TOP 10)")
        # 새로운 Figure 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 중요도 순으로 막대 그래프 출력
        # shap_values[0]은 현재 선택된 데이터 1건에 대한 영향력입니다.
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[0], 
            feature_names=all_feature_names,
            max_display=10,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf() # 그래프 초기화

    with col2:
        st.write("#### 2. 원본 피처 데이터 정보")
        # GNN 임베딩을 제외한 원본 피처 10개만 따로 보여주기
        # all_feature_names의 마지막 10개가 원본 피처명이라고 가정
        orig_feature_names = all_feature_names[-10:]
        orig_df = pd.DataFrame(node_orig, columns=orig_feature_names)
        st.table(orig_df.T.rename(columns={0: "값"}))

    st.success(f"""
    **💡 분석 결과 요약:**
    - 현재 이 거래는 **{prob:.2%}**의 확률로 사기일 가능성이 있습니다.
    - 위 그래프에서 **오른쪽(분홍색/양수)** 막대는 자금 세탁 경로 확률을 **증가**시키는 요인입니다.
    - **왼쪽(하늘색/음수)** 막대는 자금 세탁 확률을 **감소**시키는 요인입니다.
    - `GNN_Emb`로 시작하는 항목은 인접한 노드(거래처)와의 관계에서 도출된 특징입니다.
    """)
    
else:
    st.write("👈 왼쪽 사이드바에서 노드를 선택하고 분석 버튼을 눌러주세요.")
