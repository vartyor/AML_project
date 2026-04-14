"""
app.py
-------
자금 세탁 거점 분석 대시보드 — Streamlit 메인 진입점.

각 기능 모듈을 조합하여 UI를 구성합니다:
    loaders/resource_loader.py       → 모델 및 데이터 로드
    analysis/shap_analyzer.py        → SHAP 해석
    reporters/ai_report_generator.py → SAR JSON 조립 + Ollama 호출
    reporters/pdf_report_generator.py→ PDF 생성 (TTF 없이 내장 폰트 사용)

실행:
    streamlit run app.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from analysis.shap_analyzer import ShapAnalyzer
from loaders.resource_loader import (
    get_all_embeddings_cached,
    get_sorted_test_nodes,
    load_resources,
)
from reporters.ai_report_generator import build_sar_payload, generate_ai_report
from reporters.pdf_report_generator import create_pdf_report

# ======================================================================
# 0. 페이지 설정
# ======================================================================
st.set_page_config(page_title="자금 세탁 경로 탐지 시스템", layout="wide")
st.title("🛡️ 자금 세탁 거점 분석 대시보드")

# ======================================================================
# 1. 리소스 로드 (캐싱)
# ======================================================================
graph_dict, extractor, xgb_model, all_feature_names = load_resources()
all_embs_cached = get_all_embeddings_cached(
    extractor, graph_dict["x"], graph_dict["edge_index"]
)
risk_df = get_sorted_test_nodes(
    all_embs_cached, graph_dict, xgb_model, all_feature_names
)
high_risk_indices = risk_df["node_idx"].tolist()

# ======================================================================
# 2. 세션 상태 초기화
# ======================================================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "ai_report_content" not in st.session_state:
    st.session_state.ai_report_content = None
if "current_node" not in st.session_state:
    st.session_state.current_node = None

# ======================================================================
# 3. 사이드바 — 노드 선택
# ======================================================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.info("💡 아래 리스트는 모델이 판단한 **사기 의심 확률이 높은 순**으로 정렬되어 있습니다.")

    def format_node_label(idx: int) -> str:
        row = risk_df[risk_df["node_idx"] == idx].iloc[0]
        return f"ID: {idx} (위험도: {row['fraud_prob']:.1%})"

    selected_idx = st.selectbox(
        "분석할 거래(노드) 선택",
        high_risk_indices,
        format_func=format_node_label,
    )

    if selected_idx != st.session_state.current_node:
        st.session_state.analysis_done = False
        st.session_state.ai_report_content = None
        st.session_state.current_node = selected_idx

    analyze_btn = st.button("🔍 상세 분석 실행")

if analyze_btn:
    st.session_state.analysis_done = True

# ======================================================================
# 4. 메인 분석 영역
# ======================================================================
if not st.session_state.analysis_done:
    st.write(
        "👈 왼쪽 사이드바에서 노드를 선택하고 분석 버튼을 눌러주세요. "
        "(리스트 상단에 위험 계좌가 배치되어 있습니다)"
    )
    st.stop()

with st.spinner("분석 중......"):
    # 4-1. 임베딩 인덱싱 + XGBoost 예측
    node_emb = all_embs_cached[selected_idx].reshape(1, -1)
    node_orig = graph_dict["x"][selected_idx][-10:].reshape(1, -1).numpy()
    X_input = np.hstack([node_emb, node_orig])
    X_input_df = pd.DataFrame(X_input, columns=all_feature_names)
    prob = xgb_model.predict_proba(X_input)[0][1]

    # 4-2. SHAP 분석
    analyzer = ShapAnalyzer(xgb_model)
    analyzer.build_explainer(X_input)
    if analyzer.is_kernel:
        st.warning("TreeExplainer 초기화 실패 — KernelExplainer로 대체 실행 중입니다.")
    shap_values = analyzer.compute_shap_values(X_input)

    # 4-3. 메트릭 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("선택된 노드", selected_idx)
    c2.metric("자금 세탁 통로 의심 점수", f"{prob:.2%}")
    if prob > 0.5:
        c3.error("🚨 고위험 계좌(자금 세탁 의심 계좌)")
        risk_level = "위험(Danger)"
    else:
        c3.success("✅ 정상 거래 계좌")
        risk_level = "안전(Safe)"

    st.divider()

    # 4-4. SHAP 시각화
    st.write("### 📊 판단 근거 분석")
    top_feature = analyzer.most_influential_feature(shap_values, all_feature_names)
    st.info(f"#### 💡 분석 핵심 요약: 가장 큰 영향을 준 요인은 **'{top_feature}'** 입니다.")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("#### 1. 피처별 기여도 (영향력 TOP 10)")
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            analyzer.expected_value,
            shap_values[0],
            feature_names=all_feature_names,
            max_display=10,
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.write("#### 2. 원본 피처 데이터 정보")
        orig_feature_names = all_feature_names[-10:]
        orig_df = pd.DataFrame(node_orig, columns=orig_feature_names)
        st.table(orig_df.T.rename(columns={0: "값"}))

    st.success(
        f"""
        **💡 분석 결과 요약:**
        - 현재 이 거래는 **{prob:.2%}**의 확률로 사기일 가능성이 있습니다.
        - 오른쪽(분홍색/양수) 막대는 사기 확률을 **증가**시키는 요인입니다.
        - 왼쪽(하늘색/음수) 막대는 사기 확률을 **감소**시키는 요인입니다.
        - GNN_Emb로 시작하는 항목은 인접한 노드(거래처)와의 관계에서 도출된 특징입니다.
        """
    )

    # 4-5. SAR 페이로드 조립
    st.divider()
    st.write("### 📝 SAR 보고서 작성을 위한 데이터 추출")
    orig_df_dict = orig_df.to_dict(orient="records")[0]
    sar_payload, sar_json_str = build_sar_payload(
        selected_idx=selected_idx,
        prob=prob,
        shap_values=shap_values,
        X_input=X_input,
        all_feature_names=all_feature_names,
        orig_df_dict=orig_df_dict,
    )

    # 4-6. AI SAR 보고서 생성
    st.divider()
    st.subheader("AI 자동 생성 의심거래보고서(SAR)")

    if st.button("🚀 AI 전문 보고서 생성(Llama 3.1)"):
        with st.spinner("AI가 분석 데이터를 검토하여 보고서를 작성 중입니다..."):
            st.session_state.ai_report_content = generate_ai_report(sar_json_str)

    # 4-7. 보고서 표시 및 다운로드
    if st.session_state.ai_report_content:
        st.markdown("---")
        formatted_report = st.session_state.ai_report_content.replace("\n", "<br>")
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 25px; border-radius: 10px;
                        border-left: 5px solid #ff4b4b; color: #1f2937; line-height: 1.6;">
                {formatted_report}
            </div>
            """,
            unsafe_allow_html=True,
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📄 보고서 다운로드(TXT)",
                data=st.session_state.ai_report_content,
                file_name=f"SAR_Report_Node_{selected_idx}.txt",
                mime="text/plain",
            )
        with dl_col2:
            pdf_result = create_pdf_report(
                report_text=st.session_state.ai_report_content,
                node_id=selected_idx,
                risk_level=risk_level,
            )
            if isinstance(pdf_result, str):  # 성공=bytes, 실패=str(에러메시지)
                st.error(pdf_result)
            else:
                st.download_button(
                    label="📕 PDF 보고서 다운로드",
                    data=pdf_result,
                    file_name=f"SAR_Report_{selected_idx}.pdf",
                    mime="application/pdf",
                )

    # 4-8. 원본 JSON 보기
    with st.expander("📝 원본 분석 JSON 데이터 보기"):
        st.code(sar_json_str, language="json")