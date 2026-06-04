"""
generation.py
-------------
Gọi Gemini API để sinh câu trả lời dựa trên chunks retrieve được (RAG).

Cấu trúc prompt:
  - SYSTEM PROMPT (cố định): định nghĩa vai trò, quy tắc trả lời, chống bịa
  - USER PROMPT (động): câu hỏi của user + các chunks context từ retrieval

SDK: google-genai (SDK mới, thống nhất - KHÔNG dùng google-generativeai cũ đã deprecated)

Setup:
  pip install google-genai
  export GEMINI_API_KEY="your-key-here"   # lấy ở https://aistudio.google.com/apikey

Usage:
  from generation import Generator
  gen = Generator()
  answer = gen.answer("Việt vị áp dụng cho thủ môn không?", chunks)
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load file .env (đặt ở gốc dự án)
load_dotenv()


# =========================================================================
# 1. CONFIG
# =========================================================================

MODEL = "gemini-2.5-flash"   # free tier, nhanh, tiếng Việt tốt
# Các lựa chọn khác:
#   gemini-2.5-flash       - mới hơn, thông minh hơn, vẫn có free tier
#   gemini-2.0-flash-lite  - nhẹ hơn, nhanh hơn nữa
#   gemini-2.5-pro         - mạnh nhất nhưng free tier giới hạn chặt


# =========================================================================
# 2. SYSTEM PROMPT (CỐ ĐỊNH)
# =========================================================================
# Đây là "tính cách" và "quy tắc" của chatbot - không đổi giữa các câu hỏi.
# Thiết kế kỹ ở đây quyết định chất lượng + độ tin cậy của câu trả lời.

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về Luật thi đấu Bóng đá của Liên đoàn Bóng đá Việt Nam (VFF), dịch từ Luật của IFAB.

NHIỆM VỤ:
Trả lời câu hỏi của người dùng về luật bóng đá, CHỈ dựa trên các trích đoạn luật được cung cấp trong phần "NGỮ CẢNH" của mỗi câu hỏi.

QUY TẮC BẮT BUỘC:
1. CHỈ trả lời dựa trên thông tin trong NGỮ CẢNH được cung cấp. TUYỆT ĐỐI KHÔNG bịa đặt hoặc dùng kiến thức ngoài ngữ cảnh.
2. Nếu NGỮ CẢNH không chứa thông tin để trả lời, hãy nói rõ: "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu luật hiện có." KHÔNG cố đoán.
3. LUÔN trích dẫn nguồn cụ thể khi trả lời, theo định dạng: (Theo Luật X - Tên luật, mục Y) hoặc (Theo Luật X, điều khoản Z).
4. Trả lời bằng tiếng Việt, rõ ràng, chính xác, đúng thuật ngữ bóng đá.
5. Nếu câu hỏi mơ hồ hoặc có nhiều cách hiểu, hãy nêu rõ và trả lời cho các trường hợp.
6. Giữ câu trả lời tập trung, không lan man. Ưu tiên chính xác hơn dài dòng.

PHONG CÁCH:
- Thân thiện nhưng chuyên nghiệp.
- Khi giải thích quy tắc phức tạp (vd việt vị, phạt đền), có thể dùng ví dụ minh họa NHƯNG phải dựa trên nội dung trong ngữ cảnh.
- Không dùng emoji."""


# =========================================================================
# 3. USER PROMPT BUILDER (ĐỘNG)
# =========================================================================

def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Ghép câu hỏi user + các chunks context thành 1 user prompt.

    Args:
        question: câu hỏi của user
        chunks: list dicts từ retrieval.search(), mỗi dict có 'text', 'breadcrumb', ...

    Returns:
        Chuỗi user prompt hoàn chỉnh.
    """
    # Ghép các chunks thành phần NGỮ CẢNH, đánh số để LLM dễ tham chiếu
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Mỗi chunk đã có breadcrumb trong text, nhưng thêm số thứ tự cho rõ
        breadcrumb = chunk.get("breadcrumb", "")
        text = chunk.get("text", "")
        # Bỏ phần breadcrumb lặp trong text nếu có (vì sẽ hiển thị riêng)
        context_parts.append(f"[Trích đoạn {i}] {text}")

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""NGỮ CẢNH (các trích đoạn luật liên quan):

{context}

---

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

Hãy trả lời câu hỏi dựa trên NGỮ CẢNH ở trên, kèm trích dẫn nguồn."""

    return user_prompt


# =========================================================================
# 4. GENERATOR CLASS
# =========================================================================

@dataclass
class GenerationResult:
    """Kết quả sinh câu trả lời."""
    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sources: list = None        # list chunk_id đã dùng


class Generator:
    """Wrapper gọi Gemini API cho RAG."""

    def __init__(self, api_key: str | None = None, model: str = MODEL):
        """
        Args:
            api_key: Gemini API key. Nếu None, lấy từ env GEMINI_API_KEY.
            model: tên model Gemini.
        """
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Chưa có GEMINI_API_KEY. Set bằng:\n"
                "  export GEMINI_API_KEY='your-key'\n"
                "Lấy key tại: https://aistudio.google.com/apikey"
            )
        # Khởi tạo client với SDK mới
        self.client = genai.Client(api_key=key)
        self.model = model

    def answer(self,
               question: str,
               chunks: list[dict],
               temperature: float = 0.2) -> GenerationResult:
        """Sinh câu trả lời từ câu hỏi + chunks.

        Args:
            question: câu hỏi user
            chunks: list chunks từ retrieval
            temperature: độ "sáng tạo" (0.0-1.0). Với QA luật → để thấp (0.2)
                         để bám sát ngữ cảnh, ít bịa.

        Returns:
            GenerationResult với answer + metadata
        """
        # Xử lý trường hợp không có chunk nào
        if not chunks:
            return GenerationResult(
                answer="Tôi không tìm thấy thông tin liên quan trong tài liệu luật để trả lời câu hỏi này.",
                model=self.model,
                sources=[],
            )

        # Build user prompt
        user_prompt = build_user_prompt(question, chunks)

        # Gọi Gemini API
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,   # ← system prompt cố định
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )

        # Lấy token usage nếu có
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        return GenerationResult(
            answer=response.text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            sources=[c.get("chunk_id") for c in chunks],
        )

    def answer_stream(self, question: str, chunks: list[dict], temperature: float = 0.2):
        """Sinh câu trả lời dạng STREAMING (chữ hiện dần).

        Yield từng đoạn text khi model sinh ra. Dùng cho UI streaming.

        Usage:
            for piece in gen.answer_stream(question, chunks):
                print(piece, end="", flush=True)
        """
        if not chunks:
            yield "Tôi không tìm thấy thông tin liên quan trong tài liệu luật để trả lời câu hỏi này."
            return

        user_prompt = build_user_prompt(question, chunks)

        # generate_content_stream cho streaming
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text


# =========================================================================
# 5. DEMO khi chạy trực tiếp
# =========================================================================

if __name__ == "__main__":
    import sys

    # Demo: kết hợp retrieval + generation thành 1 pipeline RAG hoàn chỉnh
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval import Retriever

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/chroma_db"

    print("=" * 70)
    print("🚀 RAG PIPELINE - Retrieval + Gemini Generation")
    print("=" * 70)

    # Khởi tạo (1 lần)
    retriever = Retriever(db_path=db_path)
    generator = Generator()
    print(f"   ✅ Gemini model: {generator.model}")

    print("\n" + "=" * 70)
    print("💬 Hỏi đáp - gõ câu hỏi, Ctrl+C để thoát")
    print("=" * 70)

    while True:
        try:
            question = input("\n❓ Câu hỏi: ").strip()
            if not question:
                continue

            # BƯỚC 1: Retrieve chunks liên quan
            print("\n🔍 Đang tìm trong tài liệu...")
            chunks = retriever.search(question, top_k=5)

            print(f"   Tìm thấy {len(chunks)} trích đoạn liên quan:")
            for c in chunks:
                print(f"     - {c['chunk_id']} (sim={c['similarity']:.3f})")

            # BƯỚC 2: Generate câu trả lời (streaming)
            print("\n💡 Trả lời:\n")
            for piece in generator.answer_stream(question, chunks):
                print(piece, end="", flush=True)
            print()  # newline cuối

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break