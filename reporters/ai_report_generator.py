"""
reporters/ai_report_generator.py
----------------------------------
Ollama(llama3.1) 기반 의심거래보고서(SAR) 생성 모듈.

- generate_ai_report() : Ollama API 호출 → 보고서 문자열 반환
- build_sar_payload()  : SHAP 분석 결과를 SAR 전송용 JSON으로 조립
"""

import json
from datetime import datetime
from typing import Any

import numpy as np
import requests


# ======================================================================
# 1. Ollama API 호출
# ======================================================================

def generate_ai_report(
    json_data: str,
    model: str = "llama3.1",
    ollama_url: str = "http://localhost:11434/api/generate",
    timeout: int = 60,
) -> str:
    """
    KoFIU AML 수사관 역할 프롬프트를 사용해 Ollama 로컬 API에
    SAR 보고서 생성을 요청합니다.

    Args:
        json_data   (str): build_sar_payload()로 생성된 JSON 문자열
        model       (str): Ollama 모델명 (default: llama3.1)
        ollama_url  (str): Ollama API 엔드포인트
        timeout     (int): 요청 타임아웃 (초)

    Returns:
        str: 생성된 보고서 텍스트 (실패 시 오류 메시지)
    """
    prompt = f"""
당신은 대한민국 금융정보분석원(KoFIU) 소속의 자금세탁방지(AML) 전문 수사관입니다.
다음 데이터를 바탕으로 법적 효력을 갖출 수 있는 수준의 '의심거래보고서(SAR)'를 작성하십시오.

[작성 가이드라인]
1. 문서 번호와 보안 등급(대외비)을 포함할 것.
2. 개조식(1., 가., -)을 사용하여 명확하게 기술할 것.
3. '특정 금융거래정보의 보고 및 이용 등에 관한 법률' 제4조를 근거로 명시할 것.
4. 구성 항목:
   I.  분석 개요 (대상 계좌, 위험 등급)
   II. 주요 의심 거래 징후 (네트워크 특징, 통계적 이상치)
   III.자금세탁 위험 평가 (종합 의견)
   IV. 조치 권고 사항 (동결, 추가 조사 등)

주의: 1) 거래 패턴의 이상점 2) 법적 위반 소지 3) 위험도 판단 근거를
내부적으로 정리한 뒤, 그 내용을 바탕으로 보고서를 작성하십시오.

데이터: {json_data}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        raw = response.json().get("response", "보고서 내용 생성 실패")

        # 일본어 투 말투 보정
        raw = raw.replace("があります", "가 있습니다")
        raw = raw.replace("必要があります", "필요가 있습니다")
        return raw

    except requests.exceptions.ConnectionError:
        return "❌ Ollama 서버에 연결할 수 없습니다. localhost:11434 가 실행 중인지 확인하세요."
    except requests.exceptions.Timeout:
        return f"❌ 요청이 {timeout}초 내에 응답하지 않았습니다. 타임아웃을 늘리거나 모델을 확인하세요."
    except Exception as exc:
        return f"❌ 알 수 없는 오류: {exc}"


# ======================================================================
# 2. SAR 페이로드 조립
# ======================================================================

def build_sar_payload(
    selected_idx: int,
    prob: float,
    shap_values: np.ndarray,
    X_input: np.ndarray,
    all_feature_names: list[str],
    orig_df_dict: dict[str, Any],
) -> tuple[dict, str]:
    """
    SHAP 분석 결과와 예측 확률을 바탕으로 Ollama 전송용 SAR 데이터셋을
    구성합니다.

    Args:
        selected_idx      (int)  : 분석 대상 노드 인덱스
        prob              (float): XGBoost 예측 사기 확률
        shap_values       (ndarray): SHAP 값 배열 [1, n_features]
        X_input           (ndarray): 모델 입력 배열 [1, n_features]
        all_feature_names (list) : 전체 피처명 리스트
        orig_df_dict      (dict) : 원본 피처 10개의 {feature: value} 딕셔너리

    Returns:
        tuple:
            sar_payload (dict): 구조화된 SAR 데이터
            sar_json_str (str): JSON 직렬화 문자열
    """
    # 영향도 상위 5개 피처 추출
    top_indices = np.argsort(np.abs(shap_values[0]))[-5:][::-1]
    top_features_summary = []

    for idx in top_indices:
        feature_name = all_feature_names[idx]
        feature_val = float(X_input[0, idx])
        shap_val = float(shap_values[0][idx])

        f_type = (
            "네트워크 관계 특징"
            if feature_name.startswith("GNN_Emb")
            else "개별 거래 특징"
        )
        direction = "위험 증가(사기 의심)" if shap_val > 0 else "위험 감소(정상 의심)"

        top_features_summary.append(
            {
                "특징명": feature_name,
                "특징유형": f_type,
                "현재값": round(feature_val, 4),
                "영향도": direction,
            }
        )

    # 위험 등급 문자열
    if prob > 0.7:
        risk_level_str = "고위험(High)"
    elif prob > 0.4:
        risk_level_str = "중위험(Medium)"
    else:
        risk_level_str = "저위험(Low)"

    sar_payload = {
        "report_context": {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "target_node_id": int(selected_idx),
            "fraud_probability": f"{prob:.2%}",
            "risk_level": risk_level_str,
        },
        "key_risk_factors": top_features_summary,
        "raw_feature_data": orig_df_dict,
        "model_explanation": (
            "이 모델은 거래처 간의 송금 네트워크(GNN)와 "
            "해당 계좌의 통계적 특징(XGBoost)을 결합하여 분석한 결과입니다."
        ),
    }

    sar_json_str = json.dumps(sar_payload, indent=2, ensure_ascii=False)
    return sar_payload, sar_json_str
