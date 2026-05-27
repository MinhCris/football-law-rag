"""
chunking_metadata.py
--------------------
Chunking thuần theo METADATA STRUCTURE của PDF.

Khác với chunking thường (cắt theo regex content) hoặc fixed-size (cắt theo
số ký tự), file này dùng các DẤU HIỆU METADATA của PDF để xác định cấu trúc:

  - Font name        (vd: UTMNokiaStandardBold → LUẬT header)
  - Font size        (vd: 14pt → LUẬT, 13pt bold → Mục)
  - Font weight      (bold/regular qua flag)
  - Vị trí trang     (page number)
  - Bounding box     (tọa độ trên trang)

Triết lý:
  Tài liệu được render với font/size khác nhau cho từng cấp heading.
  Cấu trúc đó CHÍNH LÀ metadata. Ta tận dụng nó làm RANH GIỚI chunk,
  thay vì pattern-match nội dung text.

Pipeline:
  1. Extract spans (text + font + size + position) bằng PyMuPDF
  2. Học style profile: font/size nào ứng với cấp nào (LUẬT/Mục/Nội dung)
  3. Group spans thành lines (cùng dòng visual)
  4. Classify mỗi line: HEADING_L1 / HEADING_L2 / HEADING_L3 / CONTENT
  5. Build hierarchy: dùng heading levels làm ranh giới chunks
  6. Mỗi chunk = nội dung giữa 2 heading × metadata vị trí trong cây

LIMITATION (quan trọng):
  Phương pháp này chỉ detect được những cấp heading có VISUAL DISTINCTION
  (font/size/bold khác body). Nếu PDF dùng cùng style cho nội dung và một
  cấp sub-heading nào đó (vd "1.1.", "1.2." cùng font/size với body),
  cấp đó SẼ KHÔNG được detect. Đó là hạn chế cố hữu của metadata-based
  chunking - phải chấp nhận chunks to hơn, hoặc dùng hybrid với regex.

Usage:
  python chunking_metadata.py <input_pdf> <output_json>
"""

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import pymupdf


# =========================================================================
# 1. DATA MODELS
# =========================================================================

@dataclass
class Span:
    """1 span = 1 đoạn text có cùng font/size/style trên cùng 1 dòng."""
    text: str
    font: str
    size: float
    flags: int           # bit flags: bold=16, italic=2, ...
    bbox: tuple          # (x0, y0, x1, y1)
    page: int

    @property
    def is_bold(self) -> bool:
        # Trong PyMuPDF: flag 16 = bold, hoặc tên font chứa "Bold"
        return bool(self.flags & 16) or "Bold" in self.font

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2) or "Italic" in self.font


@dataclass
class Line:
    """1 line = các spans gần nhau theo trục y, ghép lại thành 1 dòng visual."""
    text: str
    spans: list           # list of Span
    page: int
    y_pos: float          # y trung bình
    dominant_font: str    # font chính (chiếm nhiều ký tự nhất)
    dominant_size: float
    is_bold: bool


@dataclass
class Heading:
    """1 heading detect được trong tài liệu."""
    level: int            # 1 (cao nhất - LUẬT), 2 (mục), 3 (điều khoản)
    text: str
    page: int
    line_index: int       # vị trí trong dãy lines toàn cục


@dataclass
class Chunk:
    """1 chunk - đơn vị retrieval."""
    chunk_id: str
    text: str             # nội dung có breadcrumb prepend
    level: int            # cấp độ trong cây
    heading_path: list    # path các heading cha (breadcrumb dạng list)
    breadcrumb: str       # path dạng string "A > B > C"
    page_start: int
    page_end: int
    char_count: int
    # Style metadata của heading (debug/verify)
    heading_font: Optional[str] = None
    heading_size: Optional[float] = None


# =========================================================================
# 2. EXTRACT SPANS từ PDF
# =========================================================================

def extract_spans(pdf_path: str, skip_first_pages: int = 4) -> list[Span]:
    """Extract toàn bộ spans (text+font+style+position) từ PDF.
    
    Đây là input thô cho mọi bước phân tích sau. Mỗi span có đủ thông tin
    để biết "text này render bằng font gì, size mấy, ở đâu trên trang".
    """
    doc = pymupdf.open(pdf_path)
    spans = []
    for page_idx, page in enumerate(doc):
        if page_idx < skip_first_pages:
            continue
        page_num = page_idx + 1
        # get_text("dict") trả về structure phân cấp: blocks > lines > spans
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 = text block, 1 = image
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        continue
                    spans.append(Span(
                        text=text,
                        font=span["font"],
                        size=round(span["size"], 1),
                        flags=span["flags"],
                        bbox=tuple(span["bbox"]),
                        page=page_num,
                    ))
    doc.close()
    return spans


# =========================================================================
# 3. STYLE PROFILER - học tự động style của từng cấp
# =========================================================================

@dataclass
class StyleProfile:
    """Profile các style đặc trưng của tài liệu.
    
    Khám phá tự động từ font/size distribution thay vì hardcode.
    Giúp script này có thể áp dụng cho PDF khác nhau mà không cần sửa code.
    """
    body_font: str           # font phổ biến nhất (nội dung)
    body_size: float
    heading_styles: list     # [(font, size, is_bold)] sắp theo cấp (cao → thấp)
    noise_styles: set        # các style "rác": header trang, số trang...


def profile_styles(spans: list[Span]) -> StyleProfile:
    """Học tự động style profile của PDF.
    
    Logic:
      - Style chiếm nhiều ký tự nhất = body style (nội dung thường)
      - Các style bold/larger size = candidate heading
      - Style xuất hiện đúng 1 lần/trang ở cùng vị trí y = header/footer (noise)
    """
    # Đếm ký tự cho mỗi (font, size, is_bold) combo
    style_chars = Counter()
    style_occurrences = Counter()  # số lần xuất hiện
    for s in spans:
        key = (s.font, s.size, s.is_bold)
        style_chars[key] += len(s.text)
        style_occurrences[key] += 1

    # Body style = chiếm nhiều ký tự nhất
    body_key = style_chars.most_common(1)[0][0]
    body_font, body_size, _ = body_key

    # Candidate heading: bold + size >= body_size, KHÔNG phải body, KHÔNG quá hiếm/quá nhiều
    candidates = []
    for (font, size, is_bold), char_count in style_chars.items():
        if (font, size, is_bold) == body_key:
            continue
        occurrences = style_occurrences[(font, size, is_bold)]
        # Heuristic noise: nếu xuất hiện ít hơn 5 lần → có thể là rác đặc biệt
        if occurrences < 5:
            continue
        # Heading thường là bold HOẶC size lớn hơn body
        if is_bold or size > body_size:
            candidates.append({
                "font": font,
                "size": size,
                "is_bold": is_bold,
                "char_count": char_count,
                "occurrences": occurrences,
            })

    # Sắp theo size giảm dần (size lớn = cấp cao). Nếu size bằng nhau,
    # font có ít occurrences hơn = cấp cao hơn (vì heading cấp cao ít xuất hiện)
    candidates.sort(key=lambda c: (-c["size"], c["occurrences"]))

    # Lấy tối đa 3 cấp heading (đủ cho hầu hết tài liệu)
    heading_styles = [
        (c["font"], c["size"], c["is_bold"])
        for c in candidates[:3]
    ]

    # Detect noise: style nhỏ (size < body_size) và bold
    # Trong PDF VFF: RobotoSlab-Bold size 9.0 là header trang lặp lại
    noise_styles = set()
    for (font, size, is_bold), occurrences in style_occurrences.items():
        if size < body_size and is_bold and occurrences > 50:
            noise_styles.add((font, size, is_bold))

    return StyleProfile(
        body_font=body_font,
        body_size=body_size,
        heading_styles=heading_styles,
        noise_styles=noise_styles,
    )


# =========================================================================
# 4. GROUP SPANS → LINES
# =========================================================================

def group_spans_to_lines(spans: list[Span], y_tolerance: float = 2.0) -> list[Line]:
    """Gom các spans cùng dòng (gần nhau theo trục y) thành 1 line.
    
    Lý do: PyMuPDF đôi khi tách 1 dòng thành nhiều spans (vd phần in đậm
    và phần thường trong cùng dòng). Cần ghép lại để classify đúng.
    """
    lines = []
    # Group theo (page, y bucket)
    spans_by_page = {}
    for s in spans:
        spans_by_page.setdefault(s.page, []).append(s)

    for page in sorted(spans_by_page.keys()):
        page_spans = sorted(spans_by_page[page], key=lambda s: (s.bbox[1], s.bbox[0]))
        # Group theo y position (cùng y ± tolerance = cùng dòng)
        current_line_spans = []
        current_y = None
        for s in page_spans:
            span_y = (s.bbox[1] + s.bbox[3]) / 2
            if current_y is None or abs(span_y - current_y) <= y_tolerance:
                current_line_spans.append(s)
                if current_y is None:
                    current_y = span_y
            else:
                # Flush line cũ
                if current_line_spans:
                    lines.append(_build_line(current_line_spans, page))
                current_line_spans = [s]
                current_y = span_y
        if current_line_spans:
            lines.append(_build_line(current_line_spans, page))
    return lines


def _build_line(spans: list[Span], page: int) -> Line:
    """Tạo Line object từ list spans cùng dòng."""
    # Sort spans theo x để text đúng thứ tự
    spans = sorted(spans, key=lambda s: s.bbox[0])
    text = "".join(s.text for s in spans).strip()
    text = re.sub(r"\s+", " ", text)

    # Dominant style = style chiếm nhiều ký tự nhất
    style_chars = Counter()
    for s in spans:
        style_chars[(s.font, s.size, s.is_bold)] += len(s.text)
    (dominant_font, dominant_size, is_bold), _ = style_chars.most_common(1)[0]

    y_pos = sum((s.bbox[1] + s.bbox[3]) / 2 for s in spans) / len(spans)

    return Line(
        text=text,
        spans=spans,
        page=page,
        y_pos=y_pos,
        dominant_font=dominant_font,
        dominant_size=dominant_size,
        is_bold=is_bold,
    )


# =========================================================================
# 5. CLASSIFY LINES BY STYLE
# =========================================================================

def classify_line_level(line: Line, profile: StyleProfile) -> int:
    """Trả về cấp heading của line, hoặc 0 nếu là content thường.
    
    Cấp 1 = LUẬT (cao nhất)
    Cấp 2 = Mục
    Cấp 3 = Điều khoản / sub-section
    0     = Nội dung thường
    """
    style_key = (line.dominant_font, line.dominant_size, line.is_bold)

    # Noise (header trang, số trang...) → không phải heading
    if style_key in profile.noise_styles:
        return -1  # đánh dấu để skip luôn

    # Match với heading styles đã profile (theo thứ tự cấp)
    for level, (font, size, is_bold) in enumerate(profile.heading_styles, 1):
        if line.dominant_font == font and line.dominant_size == size and line.is_bold == is_bold:
            return level

    return 0  # content


# =========================================================================
# 6. BUILD CHUNKS DỰA TRÊN HEADING HIERARCHY
# =========================================================================

@dataclass
class _BuildState:
    """State khi đi qua dãy lines để build chunks."""
    heading_stack: list = field(default_factory=list)
    # Stack hiện tại: [(level, text, page)] - heading từ cao xuống thấp
    buffer: list = field(default_factory=list)   # các Line đang gom
    chunk_page_start: int = 1
    chunk_heading_font: Optional[str] = None
    chunk_heading_size: Optional[float] = None


def build_chunks(lines: list[Line], profile: StyleProfile) -> list[Chunk]:
    """Build chunks dựa trên cấu trúc heading detect được.
    
    Logic:
      - Đi qua từng line theo thứ tự
      - Nếu là heading (level >= 1):
          + Đóng chunk hiện tại (nếu có nội dung)
          + Cập nhật heading stack: pop các heading có level >= heading mới
          + Push heading mới vào stack
          + Bắt đầu chunk mới
      - Nếu là content: append vào buffer
    """
    chunks = []
    state = _BuildState()

    def flush_chunk(end_page: int):
        """Đóng chunk hiện tại nếu có nội dung."""
        if not state.buffer:
            return
        body = " ".join(l.text for l in state.buffer).strip()
        body = re.sub(r"\s+", " ", body)
        if len(body) < 10:
            state.buffer = []
            return

        # Build breadcrumb từ heading stack
        if state.heading_stack:
            heading_path = [h["text"] for h in state.heading_stack]
            breadcrumb = " > ".join(heading_path)
            level = state.heading_stack[-1]["level"]
            page_start = state.heading_stack[-1]["page"]
        else:
            heading_path = ["(Không có heading)"]
            breadcrumb = "(Không có heading)"
            level = 0
            page_start = state.chunk_page_start

        # Build chunk_id từ breadcrumb (slugify đơn giản)
        chunk_id = _make_chunk_id(heading_path, page_start, len(chunks))

        # Prepend breadcrumb vào text → self-contained
        final_text = f"[{breadcrumb}]\n\n{body}"

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=final_text,
            level=level,
            heading_path=heading_path,
            breadcrumb=breadcrumb,
            page_start=page_start,
            page_end=end_page,
            char_count=len(final_text),
            heading_font=state.chunk_heading_font,
            heading_size=state.chunk_heading_size,
        ))
        state.buffer = []

    for line in lines:
        level = classify_line_level(line, profile)

        # Bỏ qua noise (header/footer trang)
        if level == -1:
            continue

        # Bỏ qua line chỉ là số (số trang)
        if re.fullmatch(r"\d{1,3}", line.text):
            continue

        if level >= 1:
            # === LÀ HEADING ===
            # Đóng chunk hiện tại trước
            flush_chunk(line.page)

            # Cập nhật heading stack: pop các heading có level >= heading mới
            # Vd: đang có [LUẬT, Mục, Điều], gặp Mục mới → pop Mục+Điều, push Mục
            while state.heading_stack and state.heading_stack[-1]["level"] >= level:
                state.heading_stack.pop()
            state.heading_stack.append({
                "level": level,
                "text": line.text,
                "page": line.page,
            })

            state.chunk_page_start = line.page
            state.chunk_heading_font = line.dominant_font
            state.chunk_heading_size = line.dominant_size

            # Heading bản thân không phải nội dung chunk → không push vào buffer
            continue

        # === LÀ CONTENT ===
        if not state.buffer:
            state.chunk_page_start = line.page
        state.buffer.append(line)

    # Flush chunk cuối
    if state.buffer:
        last_page = state.buffer[-1].page
        flush_chunk(last_page)

    return chunks


def _make_chunk_id(heading_path: list[str], page: int, seq: int) -> str:
    """Tạo chunk_id từ heading path. Slugify đơn giản."""
    if not heading_path or heading_path == ["(Không có heading)"]:
        return f"chunk_p{page}_{seq:03d}"

    # Lấy 2 heading cuối (gần nhất) để làm id - đủ ngắn nhưng có context
    parts = []
    for h in heading_path[-2:]:
        # Slugify: giữ chữ/số, thay ký tự khác bằng _
        slug = re.sub(r"[^\w]+", "_", h, flags=re.UNICODE)
        slug = slug.strip("_")[:30]
        parts.append(slug)
    return "_".join(parts) + f"_p{page}"


# =========================================================================
# 7. MAIN PIPELINE
# =========================================================================

def chunk_pdf_by_metadata(pdf_path: str, skip_first_pages: int = 4,
                          verbose: bool = True) -> tuple[list[Chunk], StyleProfile]:
    """Pipeline chính: PDF → chunks theo metadata structure."""
    if verbose:
        print(f"📖 Đọc PDF: {pdf_path}")

    # 1. Extract spans
    spans = extract_spans(pdf_path, skip_first_pages)
    if verbose:
        print(f"   {len(spans)} spans")

    # 2. Profile styles (học tự động cấu trúc)
    profile = profile_styles(spans)
    if verbose:
        print(f"\n📊 STYLE PROFILE (học tự động từ PDF):")
        print(f"   Body: font={profile.body_font}, size={profile.body_size}")
        for i, (font, size, bold) in enumerate(profile.heading_styles, 1):
            print(f"   Heading L{i}: font={font}, size={size}, bold={bold}")
        print(f"   Noise styles: {len(profile.noise_styles)} loại")

    # 3. Group spans → lines
    lines = group_spans_to_lines(spans)
    if verbose:
        print(f"\n📄 {len(lines)} lines")

    # 4. Build chunks
    chunks = build_chunks(lines, profile)
    if verbose:
        print(f"\n📦 {len(chunks)} chunks created")

    return chunks, profile


# =========================================================================
# 8. EXPORT & STATS
# =========================================================================

def save_chunks(chunks: list[Chunk], output_path: str):
    data = []
    for c in chunks:
        d = asdict(c)
        data.append(d)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(chunks)} chunks → {output_path}")


def print_stats(chunks: list[Chunk]):
    print(f"\n{'='*70}")
    print(f"📊 THỐNG KÊ CHUNKS")
    print(f"{'='*70}")
    print(f"Tổng: {len(chunks)} chunks")

    # By level
    by_level = Counter(c.level for c in chunks)
    print(f"\nTheo cấp heading:")
    for level in sorted(by_level.keys()):
        label = {0: "Không heading", 1: "Cấp 1 (LUẬT)",
                 2: "Cấp 2 (Mục)", 3: "Cấp 3 (Điều khoản)"}.get(level, f"Cấp {level}")
        print(f"  {label:25s}: {by_level[level]:4d} chunks")

    # By top-level heading (cấp 1)
    top_level_counter = Counter()
    for c in chunks:
        if c.heading_path:
            top = c.heading_path[0]
            top_level_counter[top] += 1
    print(f"\nTheo heading cấp cao nhất (top {min(20, len(top_level_counter))}):")
    for heading, cnt in top_level_counter.most_common(20):
        print(f"  {heading[:50]:50s}: {cnt:3d} chunks")

    # Char distribution
    lengths = [c.char_count for c in chunks]
    if lengths:
        print(f"\nĐộ dài chunks:")
        print(f"  min: {min(lengths)}, avg: {sum(lengths)//len(lengths)}, max: {max(lengths)}")
        too_short = sum(1 for x in lengths if x < 100)
        too_long = sum(1 for x in lengths if x > 2000)
        print(f"  < 100 ký tự: {too_short}, > 2000 ký tự: {too_long}")

    # Sample
    print(f"\n📄 SAMPLE - 3 chunks đầu tiên:")
    for c in chunks[:3]:
        print(f"\n--- {c.chunk_id} (trang {c.page_start}, level {c.level}) ---")
        print(c.text[:300] + ("..." if len(c.text) > 300 else ""))


# =========================================================================
# 9. CLI
# =========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunking_metadata.py <input_pdf> <output_json>")
        sys.exit(1)

    chunks, profile = chunk_pdf_by_metadata(sys.argv[1])
    save_chunks(chunks, sys.argv[2])
    print_stats(chunks)