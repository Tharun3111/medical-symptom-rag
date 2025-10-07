import pandas as pd
from typing import List, Dict
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by words.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in words
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        # Move start forward, accounting for overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(words):
            break
    
    return chunks


def create_chunks_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load cleaned data and create chunks with metadata.
    
    Returns:
        DataFrame with columns: chunk_id, title, chunk_text, source_id, url
    """
    df = pd.read_csv(csv_path)
    
    all_chunks = []
    chunk_id = 0
    
    for idx, row in df.iterrows():
        title = row['title']
        summary = row['summary']
        also_called = row['also_called'] if pd.notna(row['also_called']) else ''
        
        # Combine title and alternative names with summary for context
        full_text = f"{title}. "
        if also_called:
            full_text += f"Also known as: {also_called}. "
        full_text += summary
        
        # Create chunks
        chunks = chunk_text(full_text, chunk_size=400, overlap=50)
        
        for chunk in chunks:
            all_chunks.append({
                'chunk_id': chunk_id,
                'title': title,
                'chunk_text': chunk,
                'source_id': row['id'],
                'url': row['url']
            })
            chunk_id += 1
    
    chunks_df = pd.DataFrame(all_chunks)
    print(f"✓ Created {len(chunks_df)} chunks from {len(df)} documents")
    print(f"  Avg chunks per document: {len(chunks_df) / len(df):.1f}")
    
    return chunks_df


if __name__ == "__main__":
    # Test chunking
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'data' / 'medline_cleaned.csv'
    
    chunks_df = create_chunks_from_csv(csv_path)
    
    # Show sample
    print(f"\n📋 Sample chunks:")
    for idx, row in chunks_df.head(3).iterrows():
        print(f"\n{idx+1}. [{row['title']}]")
        print(f"   {row['chunk_text'][:200]}...")