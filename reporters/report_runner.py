from __future__ import annotations

from typing import Callable, Generator

from reporters.ai_report_generator import (
    _strip_rag_from_output,
    stream_ai_report,
    generate_ai_report,
)
from reporters.context_builder import ContextBuilder
from reporters.sar_template import assemble_sar_template


class ReportRunner:
    """
    SAR 보고서 생성 파이프라인 오케스트레이터.

    컨텍스트 조회(ContextBuilder) → LLM 스트리밍(Ollama) → 양식 조립
    (assemble_sar_template)의 3단계를 캡슐화합니다.

    Attributes:
        context_builder (ContextBuilder): RAG·GraphRAG 컨텍스트 조회 인스턴스

    Example:
        runner = ReportRunner(context_builder)

        # 스트리밍 모드 (Streamlit 권장)
        rag_ctx, graph_ctx = runner.build_contexts(sar_payload, node_idx)
        for token in runner.stream(sar_json_str, rag_ctx, graph_ctx):
            raw_text += token
        final_report = runner.finalize(raw_text, sar_payload, graph_ctx)

        # 원스텝 비스트리밍 모드 (폴백)
        final_report, rag_ctx, graph_ctx = runner.run(
            sar_json_str, sar_payload, node_idx
        )
    """

    # Ollama 기본 설정
    DEFAULT_MODEL       = "llama3.1"
    DEFAULT_OLLAMA_URL  = "http://localhost:11434/api/generate"

    def __init__(
        self,
        context_builder: ContextBuilder,
        model:      str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self.context_builder = context_builder
        self.model      = model
        self.ollama_url = ollama_url

    # ------------------------------------------------------------------
    # 1단계: 컨텍스트 조회
    # ------------------------------------------------------------------

    def build_contexts(
        self, sar_payload: dict, node_idx: int
    ) -> tuple[str, str]:
        """
        텍스트 RAG + GraphRAG 컨텍스트를 조회합니다.

        Args:
            sar_payload (dict): build_sar_payload() 반환값
            node_idx    (int) : 분석 대상 노드 인덱스

        Returns:
            tuple[str, str]: (rag_context, graph_context)
        """
        return self.context_builder.build_all(sar_payload, node_idx)

    # ------------------------------------------------------------------
    # 2단계: LLM 스트리밍
    # ------------------------------------------------------------------

    def stream(
        self,
        sar_json_str: str,
        rag_context:   str,
        graph_context: str,
    ) -> Generator[str, None, None]:
        """
        Ollama 스트리밍 API로 SAR 섹션(II~IV)을 토큰 단위로 생성합니다.

        Args:
            sar_json_str  (str): build_sar_payload() 반환 JSON 문자열
            rag_context   (str): 텍스트 RAG 컨텍스트
            graph_context (str): GraphRAG 컨텍스트

        Yields:
            str: 토큰 단위 텍스트 조각
        """
        yield from stream_ai_report(
            json_data     = sar_json_str,
            model         = self.model,
            ollama_url    = self.ollama_url,
            rag_context   = rag_context,
            graph_context = graph_context,
        )

    # ------------------------------------------------------------------
    # 3단계: RAG 에코 제거 + 양식 조립
    # ------------------------------------------------------------------

    def finalize(
        self,
        raw_text:      str,
        sar_payload:   dict,
        graph_context: str,
    ) -> str:
        """
        스트리밍으로 수집된 raw 텍스트를 정제하고 고정 SAR 양식에 조립합니다.

        처리 순서:
            1. RAG 에코 제거 (_strip_rag_from_output)
            2. 일본어 투 말투 보정
            3. assemble_sar_template() 호출 (Section V에 graph_context 삽입)

        Args:
            raw_text      (str) : 스트리밍 수집 전체 텍스트
            sar_payload   (dict): build_sar_payload() 반환값
            graph_context (str) : GraphRAG 결과 (Section V에 직접 삽입)

        Returns:
            str: 완성된 SAR 보고서 텍스트
        """
        clean = _strip_rag_from_output(raw_text)
        clean = clean.replace("があります", "가 있습니다")
        clean = clean.replace("必要があります", "필요가 있습니다")
        return assemble_sar_template(clean, sar_payload, graph_context=graph_context)

    # ------------------------------------------------------------------
    # 원스텝 비스트리밍 실행 (폴백용)
    # ------------------------------------------------------------------

    def run(
        self,
        sar_json_str: str,
        sar_payload:  dict,
        node_idx:     int,
        on_token:     Callable[[str], None] | None = None,
    ) -> tuple[str, str, str]:
        """
        컨텍스트 조회 → 스트리밍 → 조립을 원스텝으로 실행합니다.

        on_token 콜백이 제공되면 각 토큰마다 호출됩니다(Streamlit 진행 표시 등에 활용).

        Args:
            sar_json_str (str)           : SAR JSON 문자열
            sar_payload  (dict)          : SAR 페이로드
            node_idx     (int)           : 분석 대상 노드 인덱스
            on_token     (callable|None) : 토큰 수신 콜백 (선택)

        Returns:
            tuple[str, str, str]: (final_report, rag_context, graph_context)

        Raises:
            requests.exceptions.ConnectionError: Ollama 서버 미실행
        """
        rag_ctx, graph_ctx = self.build_contexts(sar_payload, node_idx)

        raw_text = ""
        for token in self.stream(sar_json_str, rag_ctx, graph_ctx):
            raw_text += token
            if on_token:
                on_token(token)

        final_report = self.finalize(raw_text, sar_payload, graph_ctx)
        return final_report, rag_ctx, graph_ctx
