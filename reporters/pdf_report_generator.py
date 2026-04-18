import logging
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

logging.getLogger("fpdf.fonts").setLevel(logging.ERROR)

# ── 폰트 파일 경로 상수 ────────────────────────────────────────────────────
FONT_REGULAR = "NanumGothic-Regular.ttf"
FONT_BOLD    = "NanumGothic-Bold.ttf"


def _check_fonts() -> str | None:
    """
    폰트 파일 존재 여부를 확인합니다.
    누락된 파일이 있으면 안내 메시지를 반환하고, 없으면 None을 반환합니다.
    """
    missing = [f for f in (FONT_REGULAR, FONT_BOLD) if not Path(f).exists()]
    if not missing:
        return None
    return (
        f"폰트 파일 누락: {', '.join(missing)}\n"
        "Google Fonts 버전을 다운로드 후 프로젝트 루트에 위치시켜 주세요.\n"
        "https://fonts.google.com/specimen/Nanum+Gothic\n"
        "필요 파일: NanumGothic-Regular.ttf, NanumGothic-Bold.ttf\n"
        "주의: 네이버 배포 NanumGothic.ttf(구버전)는 사용 불가합니다."
    )


def create_pdf_report(
    report_text: str,
    node_id: int | str,
    risk_level: str = "미분류",
) -> bytes | str:
    """
    AI가 생성한 SAR 보고서 텍스트를 받아 한글 PDF bytes를 반환합니다.

    Args:
        report_text (str)  : AI 생성 보고서 본문 (한글 포함)
        node_id (int|str)  : 분석 대상 노드 ID
        risk_level (str)   : 위험 등급 문자열 (예: "위험(Danger)")

    Returns:
        bytes : PDF 바이너리 (성공)
        str   : "PDF 생성 에러: ..." 메시지 (실패)
    """
    try:
        # 0. 폰트 파일 사전 검증
        
        font_error = _check_fonts()
        if font_error:
            return f"폰트 로드 에러: {font_error}"

        # 1. FPDF 인스턴스 생성 및 폰트 등록
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(left=15, top=15, right=15)
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_font("Nanum",  style="", fname=FONT_REGULAR)
        pdf.add_font("NanumB", style="", fname=FONT_BOLD)

        # 2. 상단 보안 헤더
        
        pdf.set_font("Nanum", size=9)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(
            0, 5,
            text="[대외비] 금융정보분석원(KoFIU) 수사 전용",
            align="R",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # 3. 제목 및 문서번호
        
        pdf.set_font("NanumB", size=20)
        pdf.cell(
            0, 15,
            text="의심거래보고서 (SAR)",
            align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        pdf.set_font("Nanum", size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(
            0, 6,
            text=f"문서번호 : FIU-{datetime.now().year}-{node_id}",
            align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

        # 4. 핵심 요약 표
        
        pdf.set_fill_color(240, 240, 240)
        table_rows = [
            ("분석 대상 계좌", str(node_id)),
            ("최종 위험 등급", risk_level),
            ("보고서 생성일", datetime.now().strftime("%Y-%m-%d")),
        ]
        for label, value in table_rows:
            pdf.set_font("NanumB", size=10)
            pdf.cell(
                80, 10,
                text=label,
                border=1, align="C", fill=True,
                new_x=XPos.RIGHT, new_y=YPos.TOP,
            )
            pdf.set_font("Nanum", size=10)
            pdf.cell(
                110, 10,
                text=value,
                border=1,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

        pdf.ln(10)

        # 5. 보고서 본문
        
        pdf.set_font("Nanum", size=11)
        pdf.multi_cell(
            0, 8,
            text=report_text,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        # 6. 하단 법적 고지
        
        pdf.ln(20)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Nanum", size=8)
        pdf.set_text_color(130, 130, 130)
        pdf.multi_cell(
            0, 5,
            text=(
                "본 보고서는 AI 기반 이상거래 탐지 시스템에 의해 생성된 기초 조사 자료이며, "
                "실제 수사 시 법적 근거로 활용될 수 있습니다."
            ),
            align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        return bytes(pdf.output())

    except Exception as exc:
        return f"PDF 생성 에러: {exc}"