"""
embedding.py
------------
Embedding chunks bằng bge-m3 và lưu vào ChromaDB.

Pipeline:
  1. Đọc chunks.json
  2. Load model bge-m3 (lần đầu chạy sẽ download ~2GB)
  3. Embed tất cả chunks → vectors 1024 chiều
  4. Lưu vào ChromaDB (file-based, persistent)
  5. Test query để verify

Sau khi chạy xong, dùng module retrieval.py để query.

Usage:
  python embedding.py <chunks.json> <chroma_db_dir>
  python embedding.py data/processed/chunks.json data/chroma_db
"""

import json
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# =========================================================================
# 1. CONFIG
# =========================================================================

MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "football_law"
# Batch size: auto-adjust theo device sau khi load model
BATCH_SIZE_CPU = 8     # CPU yếu, batch nhỏ
BATCH_SIZE_GPU = 32    # GPU mạnh, batch lớn
EMBEDDING_DIM = 1024   # bge-m3 trả về vector 1024 chiều


# =========================================================================
# 2. LOAD CHUNKS
# =========================================================================

def load_chunks(chunks_path: str) -> list[dict]:
    """Đọc file chunks.json."""
    print(f"📂 Đọc chunks từ: {chunks_path}")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   ✅ Loaded {len(chunks)} chunks")
    return chunks


# =========================================================================
# 3. LOAD MODEL
# =========================================================================

def load_model() -> SentenceTransformer:
    """Load bge-m3 model. Lần đầu sẽ download ~2GB từ HuggingFace."""
    print(f"\n🤖 Load model: {MODEL_NAME}")
    print("   ⚠️  Lần đầu chạy sẽ download model (~2GB), hãy kiên nhẫn...")
    t0 = time.time()
    
    # Auto-detect device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print("   🚀 Phát hiện NVIDIA GPU (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("   🍎 Phát hiện Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("   💻 Dùng CPU (sẽ chậm hơn GPU)")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    print(f"   ✅ Loaded model trong {time.time() - t0:.1f}s")
    print(f"   Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"   Device: {device}")
    return model


# =========================================================================
# 4. EMBED CHUNKS
# =========================================================================

def embed_chunks(model: SentenceTransformer, chunks: list[dict]) -> list[list[float]]:
    """Embed tất cả chunks bằng batches.
    
    Trả về list các vectors (mỗi vector là list 1024 floats).
    """
    # Chọn batch size theo device
    device = str(model.device)
    if device == "cpu":
        batch_size = BATCH_SIZE_CPU
        est_time = f"~{len(chunks) * 1.5 / 60:.0f} phút"
    else:
        batch_size = BATCH_SIZE_GPU
        est_time = f"~{len(chunks) * 0.3 / 60:.0f} phút"
    
    print(f"\n🔢 Embedding {len(chunks)} chunks (batch_size={batch_size}, device={device})")
    print(f"   Dự kiến: {est_time}")
    
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    
    elapsed = time.time() - t0
    print(f"\n   ✅ Embedded {len(chunks)} chunks trong {elapsed:.1f}s "
          f"({len(chunks)/elapsed:.1f} chunks/s)")
    print(f"   Shape: {embeddings.shape}")
    
    return embeddings.tolist()


# =========================================================================
# 5. SAVE TO CHROMADB
# =========================================================================

def save_to_chroma(chunks: list[dict],
                   embeddings: list[list[float]],
                   db_path: str) -> chromadb.Collection:
    """Lưu chunks + vectors vào ChromaDB persistent.
    
    Args:
        chunks: list dicts từ chunks.json
        embeddings: list vectors (mỗi vector là list 1024 floats)
        db_path: thư mục lưu DB (vd "data/chroma_db")
    """
    print(f"\n💾 Lưu vào ChromaDB: {db_path}")
    
    Path(db_path).mkdir(parents=True, exist_ok=True)
    
    # PersistentClient: tự động lưu xuống disk, tự load lại khi mở lại
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),  # tắt telemetry
    )
    
    # Xóa collection cũ nếu có (để re-run sạch sẽ)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   🗑️  Xóa collection cũ '{COLLECTION_NAME}'")
    except Exception:
        pass
    
    # Tạo collection mới
    # metadata={"hnsw:space": "cosine"} → dùng cosine similarity
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Chuẩn bị data
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    
    # ChromaDB chỉ accept metadata là dict các giá trị scalar (str, int, float, bool)
    # → Filter các field None và convert
    metadatas = []
    for c in chunks:
        meta = {}
        for key in ["section_type", "luat_so", "luat_ten", "muc_so", "muc_ten",
                    "dieu_khoan", "special_section", "source_page", "char_count",
                    "breadcrumb", "is_table"]:
            value = c.get(key)
            if value is None:
                continue  # ChromaDB không lưu None
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
        metadatas.append(meta)
    
    # Add vào collection
    print(f"   📤 Đang upsert {len(chunks)} records...")
    t0 = time.time()
    
    # ChromaDB upsert mạnh tay batch lớn có thể OOM, nên chia nhỏ
    BATCH_UPSERT = 100
    for i in range(0, len(chunks), BATCH_UPSERT):
        collection.add(
            ids=ids[i:i+BATCH_UPSERT],
            embeddings=embeddings[i:i+BATCH_UPSERT],
            documents=documents[i:i+BATCH_UPSERT],
            metadatas=metadatas[i:i+BATCH_UPSERT],
        )
    
    print(f"   ✅ Saved {collection.count()} records trong {time.time()-t0:.1f}s")
    return collection


# =========================================================================
# 6. TEST QUERY
# =========================================================================

def test_query(collection: chromadb.Collection, model: SentenceTransformer):
    """Test query để verify retrieval hoạt động đúng."""
    print(f"\n{'='*70}")
    print("🧪 TEST QUERIES")
    print('='*70)
    
    test_questions = [
        "Việt vị áp dụng cho thủ môn không?",
        "Khi nào cầu thủ bị thẻ đỏ trực tiếp?",
        "Kích thước sân thi đấu là bao nhiêu?",
        "Quy định về phạt đền như thế nào?",
        "Bóng phải có trọng lượng bao nhiêu?",
    ]
    
    for q in test_questions:
        print(f"\n🔍 Query: {q}")
        
        # Embed query
        query_embedding = model.encode([q], normalize_embeddings=True)[0].tolist()
        
        # Search top 3
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )
        
        # In kết quả
        for i, (doc_id, distance, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            # ChromaDB trả distance, similarity = 1 - distance (cho cosine)
            similarity = 1 - distance
            breadcrumb = metadata.get("breadcrumb", "?")
            print(f"   #{i+1} (sim={similarity:.3f}) {doc_id}")
            print(f"      {breadcrumb}")


# =========================================================================
# 7. MAIN
# =========================================================================

def main(chunks_path: str, db_path: str):
    # 1. Load chunks
    chunks = load_chunks(chunks_path)
    
    # 2. Load embedding model
    model = load_model()
    
    # 3. Embed
    embeddings = embed_chunks(model, chunks)
    
    # 4. Save to ChromaDB
    collection = save_to_chroma(chunks, embeddings, db_path)
    
    # 5. Test
    test_query(collection, model)
    
    print(f"\n{'='*70}")
    print("✅ HOÀN THÀNH!")
    print('='*70)
    print(f"   ChromaDB lưu tại: {db_path}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Số records: {collection.count()}")
    print(f"\n   → Để query, dùng module retrieval.py hoặc viết code:")
    print(f"     client = chromadb.PersistentClient(path='{db_path}')")
    print(f"     collection = client.get_collection('{COLLECTION_NAME}')")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python embedding.py <chunks.json> <chroma_db_dir>")
        print("Example: python embedding.py data/processed/chunks.json data/chroma_db")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2]) 