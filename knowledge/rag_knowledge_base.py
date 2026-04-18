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


# 상수

_CHUNK_SIZE    = 500   # 청크당 최대 문자 수
_CHUNK_OVERLAP = 100   # 청크 간 겹침 문자 수
_COLLECTION    = "aml_knowledge"


# KnowledgeBase 클래스
class KnowledgeBase: # Langchain 기반 RAG 지식베이스 클래스
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


    # 내부 초기화 (지연 로딩)

    def _get_embeddings(self) -> OllamaEmbeddings: #OllamaEmbeddings 인스턴스를 반환(최초 1회 생성)
        if self._embeddings is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._embeddings = OllamaEmbeddings(
                    model=self.embed_model,
                    base_url=self.ollama_base_url,
                )
        return self._embeddings

    def _get_vectorstore(self) -> LangChainChroma: # LangChain Chroma 벡터스토어를 반환(최초 1회 생성)
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


    # 텍스트 추출 (정적 메서드)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str | Path) -> tuple[str, str]: # PDF에서 텍스트 추출
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
    def extract_text_from_txt(txt_path: str | Path) -> str: # TXT 파일에서 문자 반환
        path = Path(txt_path)
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                return path.read_text(encoding=enc).strip()
            except (UnicodeDecodeError, LookupError):
                continue
        return ""


    # 공개 메서드

    def is_built(self) -> bool: # ChromaDB 컬렉션에 이미 벡터화된 문서가 있는지 확인
        try:
            return self._get_vectorstore()._collection.count() > 0
        except Exception:
            return False

    def build(self, force: bool = False) -> None: # 내 문서를 청크 단위로 벡터화해 ChromaDB에 저장
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
