"""
retrieval.py
------------
Module retrieval - query ChromaDB để tìm chunks liên quan đến câu hỏi.

Sử dụng:
    from retrieval import Retriever
    
    retriever = Retriever(db_path="data/chroma_db")
    results = retriever.search("Việt vị áp dụng cho thủ môn không?", top_k=5)
    
    for r in results:
        print(r["breadcrumb"], r["similarity"])
        print(r["text"])
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class Retriever:
    """Retriever wrapper cho ChromaDB + bge-m3."""
    
    def __init__(self,
                 db_path: str,
                 collection_name: str = "football_law",
                 model_name: str = "BAAI/bge-m3"):
        """
        Args:
            db_path: thư mục chứa ChromaDB
            collection_name: tên collection (mặc định "football_law")
            model_name: embedding model
        """
        # Auto-detect device
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"🤖 Loading model: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        
        print(f"📂 Opening ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection_name)
        print(f"   ✅ Collection has {self.collection.count()} records")
    
    def search(self,
               query: str,
               top_k: int = 10,
               filter_metadata: dict | None = None) -> list[dict]:
        """Search chunks liên quan đến query.
        
        Args:
            query: câu hỏi của user
            top_k: số kết quả trả về
            filter_metadata: dict filter theo metadata (vd {"luat_so": 11})
        
        Returns:
            List of dicts, mỗi dict có: chunk_id, text, similarity, metadata
            Sắp theo similarity giảm dần.
        """
        # Embed query
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=filter_metadata,
        )
        
        # Format output
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "similarity": 1 - results["distances"][0][i],  # cosine: 1 - distance
                "metadata": results["metadatas"][0][i],
                "breadcrumb": results["metadatas"][0][i].get("breadcrumb", ""),
            })
        return output
    
    def search_in_law(self, query: str, luat_so: int, top_k: int = 10) -> list[dict]:
        """Search trong 1 Luật cụ thể.
        
        Vd: search_in_law("vi phạm thủ môn", luat_so=12)
        """
        return self.search(query, top_k=top_k, filter_metadata={"luat_so": luat_so})
    
    def search_tables_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Chỉ search trong các chunks là bảng."""
        return self.search(query, top_k=top_k, filter_metadata={"is_table": True})


# =========================================================================
# DEMO khi chạy trực tiếp
# =========================================================================

if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/chroma_db"
    
    retriever = Retriever(db_path=db_path)
    
    print("\n" + "="*70)
    print("🧪 INTERACTIVE SEARCH - gõ câu hỏi, Ctrl+C để thoát")
    print("="*70)
    
    while True:
        try:
            query = input("\n❓ Câu hỏi: ").strip()
            if not query:
                continue
            
            results = retriever.search(query, top_k=5)
            
            print(f"\n📋 Top {len(results)} kết quả:")
            for i, r in enumerate(results, 1):
                print(f"\n#{i} (sim={r['similarity']:.3f}) {r['chunk_id']}")
                print(f"   📍 {r['breadcrumb']}")
                # In 200 ký tự đầu của text (sau khi bỏ breadcrumb)
                text_preview = r['text'].split('\n\n', 1)[-1][:200]
                print(f"   📄 {text_preview}...")
        
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break