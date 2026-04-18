from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.graph_rag import GraphRAGRetriever
    from knowledge.rag_knowledge_base import KnowledgeBase


class ContextBuilder: # 텍스트 RAG + GraphRAG 컨텍스트 조회를 담당하는 클래스

    def __init__(
        self,
        knowledge_base:  "KnowledgeBase | None",
        graph_retriever: "GraphRAGRetriever | None",
        rag_available:   bool, # 텍스트 RAG(ChromaDB) 가용 여부
        graph_available: bool, # GraphRAG(Neo4j) 가용 여부
    ) -> None:
        self._kb             = knowledge_base
        self._gr             = graph_retriever
        self.rag_available   = rag_available
        self.graph_available = graph_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_rag_context(self, sar_payload: dict) -> str: # 위험 등급, 피처 구성을 분석하여 최적의 검색 쿼리 구성
        if not self.rag_available or self._kb is None:
            return ""

        risk_level_str = sar_payload.get("report_context", {}).get("risk_level", "")
        top_factors    = sar_payload.get("key_risk_factors", [])
        has_network    = any("GNN" in f.get("특징명", "") for f in top_factors)

        query = (
            f"분산 송금 네트워크 자금세탁 의심거래 {risk_level_str} 보고 기준 및 조치"
            if has_network
            else f"자금세탁 의심거래 {risk_level_str} 보고 의무 징후 판단"
        )

        try:
            return self._kb.format_context(query=query, k=3, score_threshold=0.3)
        except Exception:
            return ""

    def build_graph_context(self, node_idx: int) -> str: # Neo4j GraphSAGE 리트리버 통해 대상 계좌의 거래 네트워크 구조 조회
        if not self.graph_available or self._gr is None:
            return ""

        try:
            return self._gr.format_context(node_idx=node_idx)
        except Exception:
            return ""

    def build_all( # 텍스트 RAG와 GraphRAG 컨텍스트 동시 조회
        self, sar_payload: dict, node_idx: int
    ) -> tuple[str, str]:
        rag_ctx   = self.build_rag_context(sar_payload)
        graph_ctx = self.build_graph_context(node_idx)
        return rag_ctx, graph_ctx
