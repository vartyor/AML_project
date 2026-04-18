import json
from datetime import datetime
from typing import Any, Generator

import numpy as np
import requests

from reporters.sar_template import SECTION_FORMAT_INSTRUCTIONS


# 피처 값 가독성 변환 유틸

def _humanize_feature_value(z_score: float, feature_name: str) -> dict:
    """
    StandardScaler로 정규화된 z-score 값을 사람이 이해하기 쉬운
    정성적 수준(level)과 설명(description)으로 변환합니다.

    GNN 임베딩 피처(GNN_Emb_*)와 개별 거래 피처를 구분하여 처리합니다.

    Args:
        z_score      (float): 정규화된 피처 값 (mean=0, std=1)
        feature_name (str)  : 피처 이름 (GNN_Emb_* 여부 판별에 사용)

    Returns:
        dict: {
            "수준"   : "매우 높음" | "높음" | "평균 이상" | "평균 이하" | "낮음" | "매우 낮음",
            "z점수"  : 원본 z-score (소수점 4자리),
            "설명"   : 피처 특성을 반영한 한국어 설명
        }
    """
    z = float(z_score)

    # ── 수준 레이블 결정 ──────────────────────────────────────────────
    if z >= 2.0:
        level = "매우 높음"
    elif z >= 1.0:
        level = "높음"
    elif z >= 0.0:
        level = "평균 이상"
    elif z >= -1.0:
        level = "평균 이하"
    elif z >= -2.0:
        level = "낮음"
    else:
        level = "매우 낮음"

    # ── GNN 임베딩: 네트워크 활성화 방향으로 설명 ────────────────────
    if feature_name.startswith("GNN_Emb"):
        if z >= 1.5:
            desc = "강한 사기 방향 활성화 (네트워크상 의심 패턴 밀집 영역)"
        elif z >= 0.5:
            desc = "중간 사기 방향 활성화 (주변 의심 계좌와 유사한 거래 패턴)"
        elif z >= -0.5:
            desc = "중립적 활성화 (정상·사기 경계 영역)"
        elif z >= -1.5:
            desc = "중간 정상 방향 활성화 (주변 정상 계좌와 유사한 거래 패턴)"
        else:
            desc = "강한 정상 방향 활성화 (네트워크상 정상 패턴 밀집 영역)"
        return {"수준": level, "z점수": round(z, 4), "설명": desc}

    # ── 개별 거래 피처: 피처명 기반 맥락 설명 ────────────────────────
    feature_context: dict[str, tuple[str, str]] = {
        # (높을 때 의미, 낮을 때 의미)
        "amount":              ("대규모 거래 금액", "소규모 거래 금액"),
        "log_amount":          ("대규모 거래 금액 (로그)", "소규모 거래 금액 (로그)"),
        "balance_mismatch":    ("잔액 불일치 심각 (자금 은닉 의심)", "잔액 정상 범위"),
        "hour_of_day_norm":    ("비정상 시간대 거래 (심야·새벽)", "일반 업무 시간대 거래"),
        "step":                ("장기 거래 기록", "초기 또는 단기 거래 기록"),
        "oldbalanceOrg":       ("송금 전 잔액 높음", "송금 전 잔액 낮음"),
        "newbalanceOrig":      ("송금 후 잔액 높음", "송금 후 잔액 낮음 (계좌 비우기 의심)"),
        "oldbalanceDest":      ("수신 전 잔액 높음", "수신 전 잔액 낮음"),
        "newbalanceDest":      ("수신 후 잔액 높음", "수신 후 잔액 낮음"),
        "isFraud":             ("사기 거래 비율 높음", "정상 거래 비율 높음"),
    }

    ctx_high, ctx_low = feature_context.get(
        feature_name,
        (f"{feature_name} 수치 높음", f"{feature_name} 수치 낮음"),
    )

    if z >= 0:
        desc = ctx_high
    else:
        desc = ctx_low

    # 강도 수식어 추가
    if abs(z) >= 2.0:
        desc = "⚠ " + desc + " (극단적 이상치)"
    elif abs(z) >= 1.0:
        desc = desc + " (주목할 수준)"

    return {"수준": level, "z점수": round(z, 4), "설명": desc}


# ======================================================================
# 내부 유틸: 프롬프트 조립 + RAG 섹션 후처리
# ======================================================================

# 스트리밍 시 RAG 에코 차단을 위한 마커
_RAG_OUTPUT_MARKER = "=== 섹션 생성 시작 ==="


def _build_prompt(
    json_data: str,
    rag_context: str = "",
    graph_context: str = "",
) -> str:
    """
    SAR 보고서 생성용 프롬프트를 조립합니다.

    LLM에게 SECTION_FORMAT_INSTRUCTIONS(##II##...##IV_END## 형식)로
    출력하도록 지시하여 assemble_sar_template()과 연동됩니다.

    Args:
        json_data     : SHAP 분석 결과 JSON 문자열
        rag_context   : KoFIU 법령·지침 텍스트 RAG 컨텍스트 (Phase 2)
        graph_context : Neo4j 거래 네트워크 GraphRAG 컨텍스트 (Phase 3)
    """
    internal_block = ""

    if rag_context.strip() or graph_context.strip():
        inner = ""

        if rag_context.strip():
            inner += (
                "【법령·지침 참조 자료 (KoFIU 공식 문서)】\n"
                "아래 법령·지침을 보고서 작성의 근거로 활용하십시오.\n\n"
                f"{rag_context}\n"
            )

        if graph_context.strip():
            if inner:
                inner += "\n"
            inner += (
                "【거래 네트워크 구조 분석 결과 (Neo4j GraphRAG)】\n"
                "아래는 해당 계좌의 실제 거래 그래프에서 도출된 네트워크 구조 정보입니다.\n"
                "II 섹션(의심 징후)과 III 섹션(위험 평가)에 구체적인 수치와 경로로 반영하십시오.\n\n"
                f"{graph_context}\n"
            )

        internal_block = (
            "\n---[내부 참조 자료 — 출력에 포함하지 마시오]---\n"
            + inner
            + "---[내부 참조 자료 끝]---\n"
        )

    return (
        "당신은 대한민국 금융정보분석원(KoFIU) 소속의 자금세탁방지(AML) 전문 수사관입니다.\n"
        "아래 분석 데이터를 검토하여 SAR 보고서의 각 섹션 내용을 작성하십시오.\n"
        + internal_block
        + "\n"
        + SECTION_FORMAT_INSTRUCTIONS
        + "\n"
        + "[분석 데이터]\n"
        + json_data
        + "\n\n"
        + _RAG_OUTPUT_MARKER
        + "\n"
    )


def _strip_rag_from_output(text: str) -> str:
    """
    LLM이 RAG 입력 섹션이나 마커를 출력에 포함한 경우 제거합니다.

    처리 순서:
    1. '=== 섹션 생성 시작 ===' 마커 이후만 추출
    2. '---[내부 참조 자료...]---' 블록 제거
    3. 첫 번째 섹션 마커(##II##) 이전의 잡음을 항상 제거
       (임계값 없이 마커 직전까지 모두 제거 → 일관된 시작점 보장)
    """
    import re as _re

    # 1) 스트리밍 시작 마커 이후만 추출
    if _RAG_OUTPUT_MARKER in text:
        text = text.split(_RAG_OUTPUT_MARKER, 1)[-1].strip()

    # 2) 내부 참조 자료 블록 제거 (변형 패턴 포함)
    noise_patterns = [
        r"---\[내부 참조 자료[^\]]*\]---",
        r"\[내부 참조 자료[^\]]*\]",
        r"---\[분석 데이터\]---",
    ]
    for pat in noise_patterns:
        text = _re.sub(pat, "", text, flags=_re.DOTALL)

    # 3) ##II## 마커 이전의 모든 잡음 제거 (임계값 제거 — 항상 적용)
    #    LLM이 서문을 얼마나 출력하든 동일한 시작점에서 파싱 시작
    first_marker_idx = text.find("##II##")
    if first_marker_idx > 0:
        text = text[first_marker_idx:]

    return text.strip()


# ======================================================================
# 1-A. Streaming API (권장 — 타임아웃 없음)
# ======================================================================

def stream_ai_report(
    json_data: str,
    model: str = "llama3.1",
    ollama_url: str = "http://localhost:11434/api/generate",
    connect_timeout: int = 10,
    rag_context: str = "",
    graph_context: str = "",
) -> Generator[str, None, None]:
    """
    Ollama Streaming API로 SAR 보고서를 토큰 단위로 생성합니다.

    stream=True 모드에서는 첫 토큰 수신 시점마다 read timeout이 초기화되므로
    llama3.1(8B)처럼 긴 응답도 실질적으로 타임아웃 없이 처리됩니다.

    Args:
        json_data       (str): SAR JSON 문자열
        model           (str): Ollama 모델명
        ollama_url      (str): Ollama API 엔드포인트
        connect_timeout (int): 서버 연결 타임아웃 (초, 기본 10)
        rag_context     (str): KoFIU 법령·지침 텍스트 RAG 컨텍스트
        graph_context   (str): Neo4j 거래 네트워크 GraphRAG 컨텍스트

    Yields:
        str: 토큰 단위 텍스트 조각

    Raises:
        requests.exceptions.ConnectionError: Ollama 서버 미실행
    """
    prompt  = _build_prompt(json_data, rag_context, graph_context)
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  True,
        # temperature=0.1: 출력 일관성 확보 (매 생성마다 동일한 마커 형식 준수)
        # seed=42: 동일 입력에 대해 재현 가능한 출력
        "options": {"temperature": 0.1, "seed": 42},
    }

    # connect_timeout=10s, read_timeout=None(무제한) — 스트리밍 핵심
    accumulated = ""
    marker_flushed = False  # '=== 보고서 시작 ===' 마커 이전 토큰 버퍼링용

    with requests.post(
        ollama_url,
        json=payload,
        stream=True,
        timeout=(connect_timeout, None),
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                data  = json.loads(raw_line)
                token = data.get("response", "")
                if token:
                    if not marker_flushed:
                        # 마커가 나올 때까지 토큰을 버퍼에 쌓음
                        accumulated += token
                        if _RAG_OUTPUT_MARKER in accumulated:
                            marker_flushed = True
                            # 마커 이후 내용만 yield
                            after_marker = accumulated.split(_RAG_OUTPUT_MARKER, 1)[1]
                            if after_marker.strip():
                                yield after_marker
                        # 마커가 없어도 버퍼가 너무 길면 그냥 내보냄 (안전장치)
                        elif len(accumulated) > 800:
                            marker_flushed = True
                            yield accumulated
                    else:
                        yield token
                if data.get("done"):
                    # 마커가 끝까지 안 나온 경우 버퍼 내용 전체 방출
                    if not marker_flushed and accumulated:
                        yield accumulated
                    break
            except json.JSONDecodeError:
                continue


# ======================================================================
# 1-B. 비스트리밍 API (하위 호환 · 폴백용)
# ======================================================================

def generate_ai_report(
    json_data: str,
    model: str = "llama3.1",
    ollama_url: str = "http://localhost:11434/api/generate",
    timeout: int = 300,
    rag_context: str = "",
    graph_context: str = "",
) -> str:
    """
    Ollama API를 단일 요청으로 호출해 SAR 보고서 전체를 반환합니다.
    (timeout 기본값 300초로 상향 조정)

    Streamlit에서는 stream_ai_report() 사용을 권장합니다.

    Args:
        json_data     (str): SAR JSON 문자열
        model         (str): Ollama 모델명 (default: llama3.1)
        ollama_url    (str): Ollama API 엔드포인트
        timeout       (int): 요청 타임아웃 (초, 기본 300)
        rag_context   (str): KoFIU 법령·지침 텍스트 RAG 컨텍스트
        graph_context (str): Neo4j 거래 네트워크 GraphRAG 컨텍스트

    Returns:
        str: 생성된 보고서 텍스트 (실패 시 오류 메시지)
    """
    prompt  = _build_prompt(json_data, rag_context, graph_context)
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.1, "seed": 42},
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        raw = response.json().get("response", "보고서 내용 생성 실패")

        # RAG 섹션 제거 + 일본어 투 말투 보정
        raw = _strip_rag_from_output(raw)
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
        feature_val  = float(X_input[0, idx])
        shap_val     = float(shap_values[0][idx])

        f_type = (
            "네트워크 관계 특징"
            if feature_name.startswith("GNN_Emb")
            else "개별 거래 특징"
        )
        direction = "위험 증가(사기 의심)" if shap_val > 0 else "위험 감소(정상 의심)"

        # z-score → 사람이 읽기 쉬운 수준/설명으로 변환
        human = _humanize_feature_value(feature_val, feature_name)

        top_features_summary.append(
            {
                "특징명":       feature_name,
                "특징유형":     f_type,
                # 수치(z점수)와 수준 레이블을 함께 표시 → LLM이 수치 근거와 의미를 동시에 파악
                "현재값":       f"{human['z점수']:+.4f}",
                "수준":         f"{human['수준']}  (z={human['z점수']:+.4f})",
                "값 설명":      human["설명"],
                "영향도":       direction,
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
