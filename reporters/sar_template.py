"""
reporters/sar_template.py
--------------------------
SAR(의심거래보고서) 고정 양식 정의 및 LLM 출력 조립 모듈.

[설계 원칙]
- 보고서 구조(헤더·섹션 제목·구분선·푸터)는 Python에서 고정 관리.
- LLM은 II·III·IV 섹션 내용만 생성하며, 섹션 마커로 구분.
- assemble_sar_template()이 LLM 출력을 파싱하여 고정 양식에 조립.
- 마커가 없거나 파싱 실패 시 폴백(전체 원문 그대로 섹션에 삽입).

[섹션 마커 규약]
    ##II##   ... ##II_END##    → II. 주요 의심 거래 징후
    ##III##  ... ##III_END##   → III. 자금세탁 위험 평가
    ##IV##   ... ##IV_END##    → IV. 조치 권고 사항
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# ======================================================================
# 고정 양식 정의
# ======================================================================

_W  = 70            # 구분선 너비
_EQ = "═" * _W      # 두꺼운 구분선
_DH = "─" * _W      # 얇은 구분선

SAR_TEMPLATE = (
    "\n"
    + _EQ + "\n"
    + "         의심거래보고서 (Suspicious Activity Report)\n"
    + _EQ + "\n"
    + "문서 번호  : SAR-{year}-{node_id:06d}\n"
    + "보안 등급  : 대외비 (CONFIDENTIAL)\n"
    + "작성 일자  : {date}\n"
    + "보고 근거  : 특정 금융거래정보의 보고 및 이용 등에 관한 법률 제4조\n"
    + _EQ + "\n"
    + "\n"
    + "I. 분석 개요 (Analysis Overview)\n"
    + _DH + "\n"
    + "□ 분석 대상 계좌 ID  : {node_id}\n"
    + "□ 자금세탁 위험 등급  : {risk_level}\n"
    + "□ 사기 의심 확률     : {fraud_prob}\n"
    + "□ 분석 기법          : GNN(GraphSAGE) + XGBoost 하이브리드 모델\n"
    + "□ 분석 일자          : {date}\n"
    + "\n"
    + "II. 주요 의심 거래 징후 (Key Suspicious Indicators)\n"
    + _DH + "\n"
    + "{section_ii}\n"
    + "\n"
    + "III. 자금세탁 위험 평가 (AML Risk Assessment)\n"
    + _DH + "\n"
    + "{section_iii}\n"
    + "\n"
    + "IV. 조치 권고 사항 (Recommended Actions)\n"
    + _DH + "\n"
    + "{section_iv}\n"
    + "\n"
    + _EQ + "\n"
    + "※ 본 보고서는 AI 기반 자동 분석 시스템에 의해 생성되었으며,\n"
    + "  담당자의 최종 검토 및 서명이 필요합니다.\n"
    + "작성 시스템  : KoFIU AML 분석 시스템 (GNN + XGBoost + RAG)\n"
    + "생성 일시    : {timestamp}\n"
    + _EQ + "\n"
)


# ======================================================================
# LLM 프롬프트용 섹션 형식 지시문
# ======================================================================

SECTION_FORMAT_INSTRUCTIONS = """
[출력 형식 — 반드시 아래 구분자를 포함하여 정확히 출력하십시오]

##II##
(II. 주요 의심 거래 징후: GNN 네트워크 특징과 통계적 이상치를 개조식으로 기술)
##II_END##

##III##
(III. 자금세탁 위험 평가: 종합 의견, 법적 위반 소지, 위험도 근거를 기술)
##III_END##

##IV##
(IV. 조치 권고 사항: 구체적 조치 사항을 번호로 기술)
##IV_END##

주의:
- 구분자(##II## 등)는 반드시 줄 처음에 단독으로 위치해야 합니다
- 구분자 외 추가 설명·인사말·서문을 출력하지 마십시오
- 각 섹션 내용은 개조식(1., 가., -)으로 작성하십시오
"""


# ======================================================================
# 파싱 및 조립 함수
# ======================================================================

def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """
    텍스트에서 start_marker ~ end_marker 사이의 내용을 추출합니다.

    Returns:
        str: 추출된 내용 (없으면 빈 문자열)
    """
    pattern = re.compile(
        re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _fallback_section(raw_text: str, section_label: str) -> str:
    """
    마커 파싱에 실패했을 때의 폴백 텍스트를 반환합니다.
    LLM 원문에서 해당 섹션 헤더 아래 내용을 찾거나, 없으면 원문 전체를 반환.
    """
    # 예: "II.", "III.", "IV." 로 시작하는 줄 이후 내용 탐색
    header_pat = re.compile(
        rf"{re.escape(section_label)}[^\n]*\n(.*?)(?=\n[IVX]{{1,3}}\.|$)",
        re.DOTALL,
    )
    m = header_pat.search(raw_text)
    if m:
        return m.group(1).strip()
    return "(AI 생성 내용 파싱 실패 — 원문을 참고하십시오)\n\n" + raw_text[:600]


def assemble_sar_template(
    raw_llm_output: str,
    sar_payload: dict[str, Any],
) -> str:
    """
    LLM이 생성한 섹션 마커 텍스트를 고정 SAR 양식에 조립합니다.

    Args:
        raw_llm_output (str)  : LLM 스트리밍/비스트리밍 전체 출력 텍스트
        sar_payload    (dict) : build_sar_payload()가 반환한 SAR 데이터

    Returns:
        str: 완성된 SAR 보고서 (고정 양식 + LLM 섹션 내용)
    """
    ctx = sar_payload.get("report_context", {})
    now = datetime.now()

    # ── 섹션 추출 ──────────────────────────────────────────────────────
    sec_ii  = _extract_section(raw_llm_output, "##II##",  "##II_END##")
    sec_iii = _extract_section(raw_llm_output, "##III##", "##III_END##")
    sec_iv  = _extract_section(raw_llm_output, "##IV##",  "##IV_END##")

    # 마커 없이 원문이 나온 경우 폴백
    if not sec_ii:
        sec_ii = _fallback_section(raw_llm_output, "II.")
    if not sec_iii:
        sec_iii = _fallback_section(raw_llm_output, "III.")
    if not sec_iv:
        sec_iv = _fallback_section(raw_llm_output, "IV.")

    # ── 양식 조립 ──────────────────────────────────────────────────────
    node_id    = ctx.get("target_node_id", 0)
    risk_level = ctx.get("risk_level", "—")
    fraud_prob = ctx.get("fraud_probability", "—")
    date_str   = ctx.get("analysis_date", now.strftime("%Y-%m-%d"))

    return SAR_TEMPLATE.format(
        year       = now.year,
        node_id    = node_id,
        date       = date_str,
        risk_level = risk_level,
        fraud_prob = fraud_prob,
        section_ii  = sec_ii,
        section_iii = sec_iii,
        section_iv  = sec_iv,
        timestamp  = now.strftime("%Y-%m-%d %H:%M:%S"),
    )
