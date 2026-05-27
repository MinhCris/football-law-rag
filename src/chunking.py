"""
chunking.py
-----------
Script chunk PDF Luật thi đấu Bóng đá VFF thành các chunks có cấu trúc.

CẢI TIẾN v3: Hỗ trợ extract TABLE
- Detect tables bằng PyMuPDF.find_tables()
- Convert table → markdown format (giữ cấu trúc)
- Skip text trong table bbox khi process line-by-line (tránh duplicate)
- Merge multi-page tables (bảng trải dài 2+ trang)
- Tạo chunk riêng cho mỗi table với section_type="bang"

Usage:
    python chunking.py <input_pdf> <output_json>
"""

import json
import re
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Optional

import pymupdf  # PyMuPDF


# =========================================================================
# 1. DATA MODELS
# =========================================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_type: str            # "luat" | "huong_dan" | "phu_luc" | "bang"
    luat_so: Optional[int]
    luat_ten: Optional[str]
    muc_so: Optional[int]
    muc_ten: Optional[str]
    dieu_khoan: Optional[str]
    special_section: Optional[str]
    source_page: int
    char_count: int
    breadcrumb: str
    is_table: bool = False        # ← Mới: đánh dấu chunk là bảng


# =========================================================================
# 2. CẤU HÌNH PARSING
# =========================================================================

LAW_HEADER = re.compile(
    r"^LUẬT\s+(\d{1,2})\s*[-–]\s*([A-ZĐÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÊỀẾỂỄỆÔỒỐỔỖỘƠỜỚỞỠỢƯỪỨỬỮỰÍÌỈĨỊÝỲỶỸỴÚÙỦŨỤÉÈẺẼẸÓÒỎÕỌ\s]+)$"
)
SECTION_L1 = re.compile(r"^(\d{1,2})\.\s+([^\d].{2,})$")
ARTICLE = re.compile(r"^(\d{1,2}\.\d{1,2})\.?\s+(.+)$")

HEADER_PATTERN = re.compile(r"^LUẬT THI ĐẤU BÓNG ĐÁ\s*$", re.MULTILINE)
PAGE_NUM_PATTERN = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)

SPECIAL_SECTION_HEADERS = {
    "GIỚI THIỆU": "Giới thiệu",
    "QUẢN LÝ THAY ĐỔI LUẬT": "Quản lý thay đổi Luật",
    "TƯƠNG LAI": "Tương lai",
    "CÁC LƯU Ý TRONG LUẬT THI ĐẤU": "Các lưu ý trong Luật",
    "ÁP DỤNG LUẬT": "Áp dụng luật",
    "CÁC SỬA ĐỔI CHUNG": "Các sửa đổi chung",
    "HƯỚNG DẪN ÁP DỤNG TRUẤT QUYỀN THI ĐẤU TẠM THỜI (SIN BINS)": "Hướng dẫn Sin Bins",
    "HƯỚNG DẪN ĐỐI VỚI SỬ DỤNG LẠI CẦU THỦ ĐÃ THAY RA": "Hướng dẫn sử dụng lại cầu thủ thay ra",
    "CÁC TRÌNH TỰ/THỦ TỤC VỀ VAR": "Trình tự VAR",
    "CHƯƠNG TRÌNH CHẤT LƯỢNG FIFA": "Chương trình chất lượng FIFA",
    "CÁC CƠ QUAN QUẢN LÝ BÓNG ĐÁ": "Các cơ quan quản lý bóng đá",
}

MAX_CHUNK_CHARS = 1500
TARGET_SPLIT_CHARS = 1000


# =========================================================================
# 3. HELPERS
# =========================================================================

def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_uppercase_vi(text: str) -> bool:
    letters_only = re.sub(r"[^\wÀ-ỹĐ]", "", text, flags=re.UNICODE)
    if not letters_only:
        return False
    return letters_only == letters_only.upper()


def detect_special_section(line: str) -> Optional[str]:
    line_clean = line.strip().rstrip(":")
    if not is_uppercase_vi(line_clean):
        return None
    line_upper = line_clean.upper()
    for header_pattern, friendly_name in SPECIAL_SECTION_HEADERS.items():
        if line_upper == header_pattern:
            return friendly_name
        if line_upper.startswith(header_pattern) and len(line_upper) - len(header_pattern) < 100:
            return friendly_name
    return None


def merge_uppercase_lines(lines: list[str]) -> list[str]:
    """Nối các dòng UPPERCASE liền kề thành 1 dòng (vd header bị PDF ngắt dòng)."""
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            merged.append(line)
            i += 1
            continue
        if is_uppercase_vi(line) and i + 1 < len(lines):
            j = i + 1
            while j < len(lines) and is_uppercase_vi(lines[j].strip()) and lines[j].strip():
                line = line + " " + lines[j].strip()
                j += 1
            merged.append(line)
            i = j
        else:
            merged.append(line)
            i += 1
    return merged


def clean_text(text: str) -> str:
    text = HEADER_PATTERN.sub("", text)
    text = PAGE_NUM_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > TARGET_SPLIT_CHARS and current:
            parts.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        parts.append(current.strip())
    return parts


# =========================================================================
# 4. TABLE PROCESSING
# =========================================================================

def clean_cell(cell) -> str:
    """Chuẩn hóa nội dung 1 cell: bỏ None, nối multiline, normalize whitespace."""
    if cell is None:
        return ""
    text = re.sub(r"\s+", " ", str(cell)).strip()
    return text


def table_to_markdown(table_data: list[list], title: Optional[str] = None,
                      header_rows: list[list] = None) -> str:
    """Convert table 2D list → markdown table.
    
    Args:
        table_data: rows of data (không bao gồm title row)
        title: tiêu đề bảng (vd "Kết quả của đá phạt đền")
        header_rows: các hàng header (nếu None thì lấy hàng đầu tiên làm header)
    """
    if not table_data:
        return ""

    # Clean tất cả cells
    cleaned = [[clean_cell(c) for c in row] for row in table_data]
    # Pad rows về cùng width
    col_count = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (col_count - len(row)) for row in cleaned]

    lines = []
    if title:
        lines.append(f"**Bảng: {title}**")
        lines.append("")

    # Determine header
    if header_rows is None:
        header = cleaned[0]
        data_rows = cleaned[1:]
    else:
        header = [clean_cell(c) for c in header_rows[0]]
        header = header + [""] * (col_count - len(header))
        data_rows = cleaned

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- "] * col_count) + "|")
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_table_rows(rows: list[list], col_count: int) -> str:
    """Render chỉ phần data rows (không có header) - dùng để append vào table chunk có sẵn."""
    cleaned = [[clean_cell(c) for c in row] for row in rows]
    cleaned = [row + [""] * (col_count - len(row)) for row in cleaned]
    return "\n".join("| " + " | ".join(row) + " |" for row in cleaned)


def extract_table_info(table, is_continuation: bool = False) -> dict:
    """Extract table thành dict với title, header, data tách biệt.
    
    Heuristic: nếu hàng đầu có các cell None (merged cell), đó là title row.
    
    Args:
        is_continuation: nếu True, KHÔNG strip header/title (bảng tiếp tục từ trang trước,
                         row đầu là data thật)
    """
    raw = table.extract()
    if not raw:
        return {"title": None, "header": [], "data": [], "col_count": 0}

    col_count = max(len(row) for row in raw)
    raw = [row + [None] * (col_count - len(row)) for row in raw]

    # Nếu là continuation: tất cả rows đều là data
    if is_continuation:
        return {
            "title": None,
            "header": [],
            "data": raw,
            "col_count": col_count,
        }

    # Detect title row
    title = None
    start_idx = 0
    if len(raw) > 1:
        first_row = raw[0]
        non_empty = [c for c in first_row if c is not None and clean_cell(c)]
        if len(non_empty) == 1 and col_count >= 2:
            title = clean_cell(non_empty[0])
            start_idx = 1

    if start_idx < len(raw):
        header = [raw[start_idx]]
        data = raw[start_idx + 1:]
    else:
        header = []
        data = []

    return {
        "title": title,
        "header": header,
        "data": data,
        "col_count": col_count,
    }


def is_in_bbox(item_bbox, table_bbox, margin: float = 2.0) -> bool:
    """Kiểm tra item bbox có nằm trong table bbox không."""
    ix0, iy0, ix1, iy1 = item_bbox[:4]
    tx0, ty0, tx1, ty1 = table_bbox
    return (ix0 >= tx0 - margin and ix1 <= tx1 + margin
            and iy0 >= ty0 - margin and iy1 <= ty1 + margin)


def get_non_table_text(page, table_bboxes: list) -> str:
    """Lấy text trên page nhưng EXCLUDE phần text nằm trong table bbox."""
    if not table_bboxes:
        return page.get_text()

    # Lấy text blocks với position
    blocks = page.get_text("blocks")
    # Filter: bỏ blocks nằm trong bất kỳ table bbox nào
    kept = []
    for b in blocks:
        bbox = b[:4]
        in_table = any(is_in_bbox(bbox, tb) for tb in table_bboxes)
        if not in_table:
            kept.append(b)
    # Sort theo position (top-down, left-right)
    kept.sort(key=lambda b: (b[1], b[0]))
    text = "\n".join(b[4] for b in kept)
    return text


# =========================================================================
# 5. CHUNK BUILDER
# =========================================================================

class ChunkBuilder:

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.cur_law_no: Optional[int] = None
        self.cur_law_name: Optional[str] = None
        self.cur_special: Optional[str] = None
        self.cur_section_type: str = "phu_luc"
        self.cur_muc_no: Optional[int] = None
        self.cur_muc_name: Optional[str] = None
        self.cur_article: Optional[str] = None
        self.buffer_lines: list[str] = []
        self.buffer_start_page: int = 1
        # Track table state
        self._last_table_meta: Optional[dict] = None  # info về table chunk gần nhất

    # -------------------------------------------
    # Build chunk ID & breadcrumb
    # -------------------------------------------
    def _build_breadcrumb(self) -> str:
        if self.cur_law_no is not None:
            parts = [f"LUẬT {self.cur_law_no} - {self.cur_law_name}"]
        elif self.cur_special:
            parts = [self.cur_special]
        else:
            parts = ["PHỤ LỤC"]
        if self.cur_muc_no:
            parts.append(f"{self.cur_muc_no}. {self.cur_muc_name}")
        if self.cur_article:
            parts.append(self.cur_article)
        return " > ".join(parts)

    def _build_chunk_id(self, suffix: str = "") -> str:
        if self.cur_law_no is not None:
            cid = f"L{self.cur_law_no:02d}"
            if self.cur_muc_no:
                cid += f"_M{self.cur_muc_no}"
            if self.cur_article:
                cid += f"_DK{self.cur_article}"
            else:
                cid += f"_p{self.buffer_start_page}"
        elif self.cur_special:
            prefix = re.sub(r"[^\w]+", "_", self.cur_special)[:25]
            cid = prefix
            if self.cur_muc_no:
                cid += f"_M{self.cur_muc_no}"
            if self.cur_article:
                cid += f"_DK{self.cur_article}"
            else:
                cid += f"_p{self.buffer_start_page}"
        else:
            cid = f"PL_p{self.buffer_start_page}"
        if suffix:
            cid += f"_{suffix}"
        return cid

    def _ensure_unique_id(self, cid: str) -> str:
        existing = {c.chunk_id for c in self.chunks}
        if cid not in existing:
            return cid
        i = 2
        while f"{cid}_v{i}" in existing:
            i += 1
        return f"{cid}_v{i}"

    # -------------------------------------------
    # Flush text buffer
    # -------------------------------------------
    def flush_chunk(self, current_page: int):
        if not self.buffer_lines:
            return
        body = " ".join(self.buffer_lines).strip()
        body = re.sub(r"\s+", " ", body)
        if len(body) < 20:
            self.buffer_lines = []
            return

        breadcrumb = self._build_breadcrumb()
        base_id = self._build_chunk_id()
        parts = split_long_text(body, MAX_CHUNK_CHARS)

        for idx, part_text in enumerate(parts):
            final_text = f"[{breadcrumb}]\n\n{part_text}"
            cid = base_id if len(parts) == 1 else f"{base_id}_part{idx+1}"
            cid = self._ensure_unique_id(cid)
            self.chunks.append(Chunk(
                chunk_id=cid,
                text=final_text,
                section_type=self.cur_section_type,
                luat_so=self.cur_law_no,
                luat_ten=self.cur_law_name,
                muc_so=self.cur_muc_no,
                muc_ten=self.cur_muc_name,
                dieu_khoan=self.cur_article,
                special_section=self.cur_special,
                source_page=self.buffer_start_page,
                char_count=len(final_text),
                breadcrumb=breadcrumb,
                is_table=False,
            ))

        self.buffer_lines = []
        self.buffer_start_page = current_page
        # Sau khi flush text chunk, reset tracking table cũ
        self._last_table_meta = None

    # -------------------------------------------
    # Add table chunk
    # -------------------------------------------
    def add_table_chunk(self, table_info: dict, page_num: int,
                        treat_as_continuation: bool = False):
        """Thêm 1 table chunk. Nếu là continuation của bảng trước → append rows."""
        if not table_info["data"] and not table_info["header"]:
            return

        # Flush text buffer trước khi xử lý table
        if self.buffer_lines:
            self.flush_chunk(page_num)

        # CASE 1: Continuation → append rows vào chunk table trước
        if treat_as_continuation and self._last_table_meta:
            last_chunk = self.chunks[-1]
            col_count = self._last_table_meta["col_count"]
            new_rows_md = render_table_rows(table_info["data"], col_count)
            new_text = last_chunk.text + "\n" + new_rows_md
            # Update chunk in place
            self.chunks[-1] = replace(
                last_chunk,
                text=new_text,
                char_count=len(new_text),
            )
            return

        # CASE 2: New table chunk
        title = table_info["title"]
        breadcrumb = self._build_breadcrumb()
        if title:
            breadcrumb_with_table = breadcrumb + f" > {title}"
        else:
            breadcrumb_with_table = breadcrumb + " > [Bảng]"

        md = table_to_markdown(
            table_info["data"],
            title=title,
            header_rows=table_info["header"],
        )
        final_text = f"[{breadcrumb_with_table}]\n\n{md}"

        cid = self._build_chunk_id(suffix=f"BANG_p{page_num}")
        cid = self._ensure_unique_id(cid)

        self.chunks.append(Chunk(
            chunk_id=cid,
            text=final_text,
            section_type="bang",
            luat_so=self.cur_law_no,
            luat_ten=self.cur_law_name,
            muc_so=self.cur_muc_no,
            muc_ten=self.cur_muc_name,
            dieu_khoan=self.cur_article,
            special_section=self.cur_special,
            source_page=page_num,
            char_count=len(final_text),
            breadcrumb=breadcrumb_with_table,
            is_table=True,
        ))
        self._last_table_meta = {
            "col_count": table_info["col_count"],
            "page_num": page_num,
        }

    # -------------------------------------------
    # Switch container
    # -------------------------------------------
    def _switch_to_law(self, law_no: int, law_name: str, page: int):
        self.flush_chunk(page)
        self.cur_law_no = law_no
        self.cur_law_name = law_name.strip().rstrip(".")
        self.cur_special = None
        self.cur_section_type = "luat"
        self.cur_muc_no = None
        self.cur_muc_name = None
        self.cur_article = None
        self.buffer_start_page = page

    def _switch_to_special(self, special_name: str, page: int):
        self.flush_chunk(page)
        self.cur_law_no = None
        self.cur_law_name = None
        self.cur_special = special_name
        self.cur_section_type = "huong_dan" if "Hướng dẫn" in special_name else "phu_luc"
        self.cur_muc_no = None
        self.cur_muc_name = None
        self.cur_article = None
        self.buffer_start_page = page

    # -------------------------------------------
    # Feed line
    # -------------------------------------------
    def feed_line(self, line: str, page_num: int):
        line = normalize_line(line)
        if not line:
            return

        m = LAW_HEADER.match(line)
        if m and is_uppercase_vi(m.group(2)):
            self._switch_to_law(int(m.group(1)), m.group(2), page_num)
            return

        special = detect_special_section(line)
        if special:
            self._switch_to_special(special, page_num)
            return

        m = ARTICLE.match(line)
        if m and (self.cur_law_no is not None or self.cur_special is not None):
            self.flush_chunk(page_num)
            self.cur_article = m.group(1)
            content = m.group(2)
            self.buffer_lines = [f"{self.cur_article}. {content}"]
            self.buffer_start_page = page_num
            return

        m = SECTION_L1.match(line)
        if m and (self.cur_law_no is not None or self.cur_special is not None):
            self.flush_chunk(page_num)
            self.cur_muc_no = int(m.group(1))
            self.cur_muc_name = m.group(2).strip().rstrip(".")
            self.cur_article = None
            self.buffer_start_page = page_num
            return

        self.buffer_lines.append(line)


# =========================================================================
# 6. MAIN PIPELINE
# =========================================================================

def chunk_pdf(pdf_path: str, skip_first_pages: int = 4) -> list[Chunk]:
    print(f"📖 Đang đọc PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    print(f"   Tổng {len(doc)} trang (bỏ qua {skip_first_pages} trang đầu)")

    builder = ChunkBuilder()
    last_page = skip_first_pages + 1
    prev_page_had_table_at_bottom = False

    for i, page in enumerate(doc):
        if i < skip_first_pages:
            continue
        page_num = i + 1
        last_page = page_num

        # 1. Tìm tables trên page
        tabs_finder = page.find_tables()
        tables = list(tabs_finder.tables)
        table_bboxes = [t.bbox for t in tables]

        # 2. Lấy text NGOÀI table
        if table_bboxes:
            raw_text = get_non_table_text(page, table_bboxes)
        else:
            raw_text = page.get_text()
        cleaned = clean_text(raw_text)
        lines = cleaned.split("\n")
        lines = merge_uppercase_lines(lines)

        # 3. Process text lines (như cũ)
        for line in lines:
            builder.feed_line(line, page_num)

        # 4. Process tables (sau khi text đã xử lý)
        for tbl_idx, tbl in enumerate(tables):
            # Heuristic continuation: page trước có table ở bottom
            # AND page này có table bắt đầu ở TOP (y < 100)
            is_continuation = (
                prev_page_had_table_at_bottom
                and tbl_idx == 0
                and tbl.bbox[1] < 100
            )
            table_info = extract_table_info(tbl, is_continuation=is_continuation)
            builder.add_table_chunk(table_info, page_num,
                                    treat_as_continuation=is_continuation)

        # 5. Update flag cho page sau
        page_height = page.rect.height
        prev_page_had_table_at_bottom = bool(tables) and any(
            t.bbox[3] > page_height - 80 for t in tables  # table chạm gần đáy
        )

    builder.flush_chunk(last_page)
    doc.close()
    return builder.chunks


def save_chunks(chunks: list[Chunk], output_path: str):
    data = [asdict(c) for c in chunks]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu {len(chunks)} chunks vào: {output_path}")


def print_stats(chunks: list[Chunk]):
    print(f"\n📊 THỐNG KÊ CHUNKS")
    print(f"   Tổng: {len(chunks)} chunks")

    by_type = {}
    for c in chunks:
        by_type[c.section_type] = by_type.get(c.section_type, 0) + 1
    print(f"   Theo loại: {by_type}")

    table_chunks = [c for c in chunks if c.is_table]
    print(f"   📋 Số bảng: {len(table_chunks)}")
    for tc in table_chunks:
        print(f"      - {tc.chunk_id} (trang {tc.source_page}, {tc.char_count} ký tự)")
        print(f"        {tc.breadcrumb}")

    by_law = {}
    law_names = {}
    for c in chunks:
        if c.luat_so:
            by_law[c.luat_so] = by_law.get(c.luat_so, 0) + 1
            if c.luat_so not in law_names or len(c.luat_ten) > len(law_names[c.luat_so]):
                law_names[c.luat_so] = c.luat_ten
    print(f"   Theo Luật:")
    for law_no in sorted(by_law.keys()):
        print(f"     Luật {law_no:2d} ({law_names[law_no]}): {by_law[law_no]} chunks")

    by_special = {}
    for c in chunks:
        if c.special_section:
            by_special[c.special_section] = by_special.get(c.special_section, 0) + 1
    if by_special:
        print(f"   Theo phần đặc biệt:")
        for name, cnt in by_special.items():
            print(f"     {name}: {cnt} chunks")

    lengths = [c.char_count for c in chunks]
    if lengths:
        print(f"   Độ dài: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)//len(lengths)}, "
              f">1500: {sum(1 for x in lengths if x > 1500)}")


# =========================================================================
# 7. CLI
# =========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python chunking.py <input_pdf> <output_json>")
        sys.exit(1)
    chunks = chunk_pdf(sys.argv[1])
    save_chunks(chunks, sys.argv[2])
    print_stats(chunks)