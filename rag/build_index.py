import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from tqdm import tqdm
from chunker import create_chunks_from_csv


def build_faiss_index(chunks_df: pd.DataFrame, model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Create embeddings and build FAISS index.
    
    Args:
        chunks_df: DataFrame with chunk_text column
        model_name: SentenceTransformer model to use
    
    Returns:
        tuple: (faiss_index, embeddings_array, model)
    """
    print(f"🤖 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Get embedding dimension
    sample_embedding = model.encode("test")
    embedding_dim = len(sample_embedding)
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Encode all chunks with progress bar
    print(f"\n📊 Creating embeddings for {len(chunks_df)} chunks...")
    texts = chunks_df['chunk_text'].tolist()
    
    # Batch encode for efficiency
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"✓ Created embeddings with shape: {embeddings.shape}")
    
    # Build FAISS index
    print(f"\n🔍 Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
    index.add(embeddings.astype('float32'))
    
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    
    return index, embeddings, model


def save_index_and_metadata(index, embeddings, chunks_df, model):
    """Save FAISS index, embeddings, and metadata."""
    store_dir = Path(__file__).parent.parent / 'store'
    store_dir.mkdir(exist_ok=True)
    
    # Save FAISS index
    index_path = store_dir / 'faiss_index.bin'
    faiss.write_index(index, str(index_path))
    print(f"✅ Saved FAISS index to: {index_path}")
    
    # Save embeddings
    embeddings_path = store_dir / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"✅ Saved embeddings to: {embeddings_path}")
    
    # Save metadata (chunks DataFrame)
    metadata_path = store_dir / 'chunks_metadata.pkl'
    chunks_df.to_pickle(metadata_path)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    # Save model name for later use
    config_path = store_dir / 'config.pkl'
    config = {'model_name': model.get_sentence_embedding_dimension()}
    with open(config_path, 'wb') as f:
        pickle.dump({'model_name': 'BAAI/bge-small-en-v1.5'}, f)
    print(f"✅ Saved config to: {config_path}")


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'data' / 'medline_cleaned.csv'
    
    # Step 1: Create chunks
    print("="*60)
    print("STEP 1: Creating text chunks")
    print("="*60)
    chunks_df = create_chunks_from_csv(csv_path)
    
    # Step 2: Build index
    print("\n" + "="*60)
    print("STEP 2: Building FAISS index")
    print("="*60)
    index, embeddings, model = build_faiss_index(chunks_df)
    
    # Step 3: Save everything
    print("\n" + "="*60)
    print("STEP 3: Saving index and metadata")
    print("="*60)
    save_index_and_metadata(index, embeddings, chunks_df, model)
    
    print("\n" + "="*60)
    print("✨ Index building complete!")
    print("="*60)
    print(f"📊 Total chunks indexed: {len(chunks_df)}")
    print(f"🔢 Embedding dimension: {embeddings.shape[1]}")