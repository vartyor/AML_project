from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from knowledge.graph_rag import GraphRAGRetriever, get_graph_retriever
from knowledge.rag_knowledge_base import KnowledgeBase

# Streamlit 캐시 초기화 함수 (앱 생명주기 동안 1회 실행)

@st.cache_resource(show_spinner="📚 KoFIU 지식베이스 로딩 중...")
def load_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase(
        pdf_directory="knowledge_base",
        persist_dir="chroma_db",
        embed_model="nomic-embed-text",
    )
    kb.build()
    return kb


@st.cache_resource(show_spinner="🔗 Neo4j 그래프 DB 연결 중...")
def load_graph_retriever() -> GraphRAGRetriever:
    return get_graph_retriever()


# ======================================================================
# 통합 초기화 유틸
# ======================================================================

@dataclass
class KnowledgeResources:
    """초기화된 지식 리소스 및 가용성 플래그를 담는 데이터 클래스."""
    knowledge_base: KnowledgeBase | None
    graph_retriever: GraphRAGRetriever | None
    rag_available: bool
    graph_available: bool
    rag_error: str          # 초기화 실패 시 오류 메시지 (성공 시 "")
    graph_error: str        # 초기화 실패 시 오류 메시지 (성공 시 "")


def init_knowledge_resources() -> KnowledgeResources:
    # ── 텍스트 RAG (KoFIU 법령·지침, Phase 2) ───────────────────────
    knowledge_base: KnowledgeBase | None = None
    rag_available  = False
    rag_error      = ""

    try:
        knowledge_base = load_knowledge_base()
        rag_available  = True
    except Exception as exc:
        rag_error = str(exc)

    # ── GraphRAG (Neo4j 거래 네트워크, Phase 3) ──────────────────────
    graph_retriever: GraphRAGRetriever | None = None
    graph_available  = False
    graph_error      = ""

    try:
        graph_retriever = load_graph_retriever()
        graph_available = graph_retriever.is_available
    except Exception as exc:
        graph_error = str(exc)

    return KnowledgeResources(
        knowledge_base  = knowledge_base,
        graph_retriever = graph_retriever,
        rag_available   = rag_available,
        graph_available = graph_available,
        rag_error       = rag_error,
        graph_error     = graph_error,
    )
