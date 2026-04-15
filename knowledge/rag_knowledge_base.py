"""
knowledge/rag_knowledge_base.py
---------------------------------
LangChain + Ollama 임베딩 + ChromaDB 기반 로컬 RAG 지식베이스 모듈.

[LangChain 스택]
  - langchain_community.embeddings.OllamaEmbeddings  → 로컬 임베딩
  - langchain_community.vectorstores.Chroma          → 벡터스토어
  - langchain_text_splitters.RecursiveCharacterTextSplitter → 청킹
  - langchain_core.documents.Document                → 문서 표현

[호환성 처리]
  - chromadb 1.5.7 호환을 위해 chromadb.PersistentClient를
    직접 생성 후 LangChain Chroma의 client= 파라미터로 전달.
    (LangChain 내부의 deprecated chromadb.Client() 호출 우회)
  - OllamaEmbeddings 는 langchain_community 0.3.1 이후 deprecated 이나
    langchain_ollama 미설치 환경에서 동작 가능한 유일한 옵션으로 사용.

[PDF 텍스트 추출 한계 및 대응]
  벡터 아웃라인 PDF(KoFIU 연차보고서 등)는 PyPDF2로 추출 불가.
  → tools/extract_pdf_to_txt.py 로 TXT 변환 후 knowledge_base/에 저장.
  → 같은 이름의 .txt 파일이 있으면 PDF보다 우선 처리.

사전 준비:
    ollama pull nomic-embed-text   ← 로컬 임베딩 모델 (최초 1회)

사용 예:
    kb = KnowledgeBase(pdf_directory="knowledge_base")
    kb.build()
    context = kb.format_context("자금세탁 의심 패턴")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import chromadb
from PyPDF2 import PdfReader

# ── LangChain imports ──────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma as LangChainChroma

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ======================================================================
# 상수
# ======================================================================
_CHUNK_SIZE    = 500   # 청크당 최대 문자 수
_CHUNK_OVERLAP = 100   # 청크 간 겹침 문자 수
_COLLECTION    = "aml_knowledge"


# ======================================================================
# KnowledgeBase 클래스
# ======================================================================

class KnowledgeBase:
    """
    LangChain 기반 RAG 지식베이스 클래스.

    Args:
        pdf_directory   (str): PDF/TXT 파일이 저장된 디렉토리 경로
        persist_dir     (str): ChromaDB 영속 저장 경로 (default: "chroma_db")
        embed_model     (str): Ollama 임베딩 모델명 (default: "nomic-embed-text")
        ollama_base_url (str): Ollama 서버 주소 (default: "http://localhost:11434")
        chunk_size      (int): 청크당 최대 문자 수 (default: 500)
        chunk_overlap   (int): 청크 간 겹침 문자 수 (default: 100)
    """

    def __init__(
        self,
        pdf_directory: str = "knowledge_base",
        persist_dir: str = "chroma_db",
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        self.pdf_directory   = Path(pdf_directory)
        self.persist_dir     = persist_dir
        self.embed_model     = embed_model
        self.ollama_base_url = ollama_base_url

        # LangChain 컴포넌트 (지연 초기화)
        self._embeddings: OllamaEmbeddings | None = None
        self._vectorstore: LangChainChroma | None = None

        # RecursiveCharacterTextSplitter — 한국어 문서에 적합한 구분자 설정
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------
    # 내부 초기화 (지연 로딩)
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> OllamaEmbeddings:
        """OllamaEmbeddings 인스턴스를 반환합니다 (최초 1회 생성)."""
        if self._embeddings is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._embeddings = OllamaEmbeddings(
                    model=self.embed_model,
                    base_url=self.ollama_base_url,
                )
        return self._embeddings

    def _get_vectorstore(self) -> LangChainChroma:
        """
        LangChain Chroma 벡터스토어를 반환합니다 (최초 1회 생성).

        [호환성 처리]
        chromadb.PersistentClient를 직접 생성하여 LangChain Chroma에 전달합니다.
        이렇게 하면 LangChain 내부에서 deprecated chromadb.Client()를 호출하는
        문제를 우회할 수 있습니다.
        """
        if self._vectorstore is None:
            chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._vectorstore = LangChainChroma(
                    collection_name=_COLLECTION,
                    embedding_function=self._get_embeddings(),
                    client=chroma_client,
                    collection_metadata={"hnsw:space": "cosine"},
                )
        return self._vectorstore

    # ------------------------------------------------------------------
    # 텍스트 추출 (정적 메서드)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text_from_pdf(pdf_path: str | Path) -> tuple[str, str]:
        """
        PDF에서 텍스트를 추출합니다 (페이지별 오류 허용).

        Returns:
            tuple[str, str]: (추출된 텍스트, 상태 메시지)
        """
        path = Path(pdf_path)
        try:
            reader       = PdfReader(str(path))
            pages        = []
            failed_pages = 0
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    failed_pages += 1
            full_text = "\n".join(pages).strip()
            status = (
                f"OK ({len(reader.pages)}p, 실패 {failed_pages}p)"
                if full_text
                else f"빈 텍스트 ({len(reader.pages)}p — 벡터 아웃라인 또는 스캔 PDF)"
            )
            return full_text, status
        except Exception as e:
            return "", f"예외: {e}"

    @staticmethod
    def extract_text_from_txt(txt_path: str | Path) -> str:
        """TXT 파일에서 텍스트를 읽어 반환합니다."""
        path = Path(txt_path)
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                return path.read_text(encoding=enc).strip()
            except (UnicodeDecodeError, LookupError):
                continue
        return ""

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def is_built(self) -> bool:
        """ChromaDB 컬렉션에 이미 벡터화된 문서가 있는지 확인합니다."""
        try:
            return self._get_vectorstore()._collection.count() > 0
        except Exception:
            return False

    def build(self, force: bool = False) -> None:
        """
        knowledge_base/ 내 문서(PDF + TXT)를 청크 단위로 벡터화하여 ChromaDB에 저장합니다.

        처리 우선순위:
          1. 같은 stem의 .txt 파일이 있으면 TXT 우선
          2. TXT 없으면 PDF → PyPDF2 추출 시도
          3. 둘 다 실패하면 건너뜀 (안내 메시지 출력)

        Args:
            force (bool): True면 기존 데이터 무시 후 재빌드
        """
        if not force and self.is_built():
            print("[KnowledgeBase] 기존 지식베이스 감지 → 재빌드 건너뜀.")
            return

        pdf_files = {p.stem: p for p in sorted(self.pdf_directory.glob("*.pdf"))}
        txt_files = {p.stem: p for p in sorted(self.pdf_directory.glob("*.txt"))}
        all_stems = sorted(set(pdf_files) | set(txt_files))

        if not all_stems:
            print(f"[KnowledgeBase] '{self.pdf_directory}' 에 처리할 문서가 없습니다.")
            return

        vectorstore  = self._get_vectorstore()
        total_chunks = 0
        ok_count     = 0
        skip_count   = 0

        for stem in all_stems:
            text   = ""
            source = ""

            # 1) TXT 우선
            if stem in txt_files:
                text   = self.extract_text_from_txt(txt_files[stem])
                source = txt_files[stem].name
                if text:
                    print(f"  📄  TXT 처리: {source}")

            # 2) PDF 폴백
            if not text and stem in pdf_files:
                pdf_text, status = self.extract_text_from_pdf(pdf_files[stem])
                source = pdf_files[stem].name
                if pdf_text:
                    text = pdf_text
                    print(f"  📑  PDF 추출: {source} [{status}]")
                else:
                    print(
                        f"  ⚠️  PDF 텍스트 추출 불가: {source} [{status}]\n"
                        f"     → tools/extract_pdf_to_txt.py 실행 후 '{stem}.txt' 생성"
                    )

            if not text:
                skip_count += 1
                continue

            # ── LangChain RecursiveCharacterTextSplitter 청킹 ─────────
            lc_docs: list[Document] = self._splitter.create_documents(
                texts=[text],
                metadatas=[{"source": source}],
            )

            # ── 청크 ID 생성 (upsert를 위해 결정적 ID 사용) ───────────
            ids = [f"{stem}__chunk_{i}" for i in range(len(lc_docs))]

            # ── LangChain Chroma add_documents ────────────────────────
            vectorstore.add_documents(documents=lc_docs, ids=ids)
            total_chunks += len(lc_docs)
            ok_count     += 1
            print(f"  ✅  저장 완료: {source}  ({len(lc_docs)} 청크)")

        print(
            f"[KnowledgeBase] 완료 — 처리 {ok_count}개 / "
            f"건너뜀 {skip_count}개 / 총 {total_chunks}개 청크"
        )

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        """
        자연어 쿼리로 유사 청크를 검색합니다.

        Returns:
            list[dict]: {"text": ..., "source": ..., "score": ...} 리스트
                        score: 0~1 (1에 가까울수록 유사)
        """
        results: list[tuple[Document, float]] = (
            self._get_vectorstore().similarity_search_with_score(query, k=k)
        )
        return [
            {
                "text":   doc.page_content,
                "source": doc.metadata.get("source", "알 수 없음"),
                # LangChain Chroma는 cosine distance 반환 → 1-dist = similarity
                "score":  round(1.0 - float(score), 4),
            }
            for doc, score in results
        ]

    def format_context(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.3,
    ) -> str:
        """
        검색 결과를 LLM 프롬프트 주입용 문자열로 포매팅합니다.

        Args:
            query           (str)  : 검색 쿼리
            k               (int)  : 최대 반환 청크 수
            score_threshold (float): 이 값 미만 청크 제외 (노이즈 방지)

        Returns:
            str: 포매팅된 컨텍스트 문자열 (결과 없으면 빈 문자열)
        """
        results  = self.search(query, k=k)
        filtered = [r for r in results if r["score"] >= score_threshold]

        if not filtered:
            return ""

        parts = []
        for i, r in enumerate(filtered, 1):
            parts.append(
                f"[참고 {i}] 출처: {r['source']} (유사도: {r['score']:.2f})\n"
                f"{r['text']}"
            )
        return "\n\n".join(parts)
