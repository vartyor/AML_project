from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# 고정 양식 정의
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
    + "V. 거래 네트워크 구조 분석 (Transaction Network Analysis — GraphRAG)\n"
    + _DH + "\n"
    + "{section_v}\n"
    + "\n"
    + _EQ + "\n"
    + "※ 본 보고서는 AI 기반 자동 분석 시스템에 의해 생성되었으며,\n"
    + "  담당자의 최종 검토 및 서명이 필요합니다.\n"
    + "작성 시스템  : KoFIU AML 분석 시스템 (GNN + XGBoost + RAG + GraphRAG)\n"
    + "생성 일시    : {timestamp}\n"
    + _EQ + "\n"
)

# Neo4j 미연결 시 Section V에 삽입할 기본 메시지
_SECTION_V_UNAVAILABLE = (
    "※ Neo4j 그래프 DB에 연결되지 않아 네트워크 구조 분석을 수행할 수 없습니다.\n"
    "  Neo4j Desktop을 실행하고 데이터를 로드한 후 보고서를 재생성하십시오."
)


# ======================================================================
# LLM 프롬프트용 섹션 형식 지시문
# ======================================================================

SECTION_FORMAT_INSTRUCTIONS = """\
[출력 형식 — 아래 규칙을 반드시 준수하십시오]

첫 번째 출력 문자는 반드시 '#' 이어야 합니다. 서문·인사말·설명을 일절 출력하지 마십시오.

##II##
(II. 주요 의심 거래 징후: GNN 네트워크 특징과 통계적 이상치를 개조식으로 기술)
##II_END##

##III##
(III. 자금세탁 위험 평가: 종합 의견, 법적 위반 소지, 위험도 근거를 기술)
##III_END##

##IV##
(IV. 조치 권고 사항: 구체적 조치 사항을 번호로 기술)
##IV_END##

[엄격 규칙]
- 구분자(##II## / ##II_END## 등)는 줄의 맨 앞에 단독으로 위치해야 합니다
- 구분자에 **, *, [], () 같은 마크다운 기호를 절대 붙이지 마십시오
- ##II## 이전에 어떠한 텍스트도 출력하지 마십시오
- 각 섹션 내용은 개조식(1., 가., -)으로 작성하십시오
"""

# 내부 유틸

def _safe_section(text: str) -> str:
    """
    str.format() 호출 시 LLM 출력의 중괄호({, })가 포맷 지시자로 해석되어
    KeyError 또는 ValueError가 발생하는 것을 방지합니다.

    Python의 str.format()은 {{ → {, }} → } 로 리터럴 변환하므로,
    LLM 출력 내의 중괄호를 이중 중괄호로 이스케이프합니다.
    """
    return text.replace("{", "{{").replace("}", "}}")


def _normalize_markers(text: str) -> str:
    """
    LLM이 섹션 마커를 마크다운 변형으로 출력한 경우 정규화합니다.

    처리 대상 변형 예시:
        **##II##**   → ##II##
        ** ##II## ** → ##II##
        ## II ##     → ##II##
        [##II##]     → ##II##
        `##II##`     → ##II##
    """
    # 1) 마커 앞뒤의 마크다운 기호 제거 (* ** ` [ ] ( ))
    text = re.sub(r'[*`\[\]()]+\s*(##\w+##)\s*[*`\[\]()]+', r'\1', text)
    # 2) ## 내부 공백 제거: ## II ## → ##II##
    text = re.sub(r'##\s+(\w+)\s+##', r'##\1##', text)
    # 3) 줄 앞쪽의 공백+마크다운 헤더(###, ##) 제거 후 마커만 남김
    #    예: "### ##II##" → "##II##"
    text = re.sub(r'^[ \t]*#{1,3}[ \t]+(##\w+##)', r'\1', text, flags=re.MULTILINE)
    return text


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:

    pattern = re.compile( # start_marker - end_marker 사이 내용 추출
        re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
        re.DOTALL | re.IGNORECASE,
    )

    # 1차 시도: 마커 정규화 후 추출 → 내용에 ** / `` 잔류 없음
    normalized = _normalize_markers(text)
    match = pattern.search(normalized)
    if match:
        return match.group(1).strip()

    # 2차 시도: 원문에서 직접 추출 (폴백)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return ""


def _fallback_section(raw_text: str, section_label: str) -> str: # 마커 파싱에 실패 시 폴백 텍스트 반환
    header_pat = re.compile(
        rf"{re.escape(section_label)}[^\n]*\n(.*?)(?=\n[IVX]{{1,3}}\.|$)",
        re.DOTALL,
    )
    m = header_pat.search(raw_text)
    if m:
        return m.group(1).strip()
    return "(AI 생성 내용 파싱 실패 — 원문을 참고하십시오)\n\n" + raw_text[:600]


# 조립 함수

def assemble_sar_template(
    raw_llm_output: str,
    sar_payload: dict[str, Any],
    graph_context: str = "",
) -> str:
    ctx = sar_payload.get("report_context", {})
    now = datetime.now()

    # ── 섹션 추출 (II·III·IV : LLM 생성) ─────────────────────────────
    sec_ii  = _extract_section(raw_llm_output, "##II##",  "##II_END##")
    sec_iii = _extract_section(raw_llm_output, "##III##", "##III_END##")
    sec_iv  = _extract_section(raw_llm_output, "##IV##",  "##IV_END##")

    # 마커 없이 원문이 나온 경우 폴백
    if not sec_ii:
        sec_ii  = _fallback_section(raw_llm_output, "II.")
    if not sec_iii:
        sec_iii = _fallback_section(raw_llm_output, "III.")
    if not sec_iv:
        sec_iv  = _fallback_section(raw_llm_output, "IV.")

    # ── Section V : GraphRAG 결과 직접 삽입 ───────────────────────────
    sec_v = graph_context.strip() if graph_context.strip() else _SECTION_V_UNAVAILABLE

    # ── str.format() 충돌 방지: 중괄호 이스케이프 ─────────────────────
    sec_ii  = _safe_section(sec_ii)
    sec_iii = _safe_section(sec_iii)
    sec_iv  = _safe_section(sec_iv)
    sec_v   = _safe_section(sec_v)

    # ── 양식 조립 ──────────────────────────────────────────────────────
    node_id    = ctx.get("target_node_id", 0)
    risk_level = ctx.get("risk_level", "—")
    fraud_prob = ctx.get("fraud_probability", "—")
    date_str   = ctx.get("analysis_date", now.strftime("%Y-%m-%d"))

    return SAR_TEMPLATE.format(
        year        = now.year,
        node_id     = node_id,
        date        = date_str,
        risk_level  = risk_level,
        fraud_prob  = fraud_prob,
        section_ii  = sec_ii,
        section_iii = sec_iii,
        section_iv  = sec_iv,
        section_v   = sec_v,
        timestamp   = now.strftime("%Y-%m-%d %H:%M:%S"),
    )
