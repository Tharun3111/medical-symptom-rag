import pandas as pd
from pathlib import Path
from data_loader import parse_medlineplus_xml


def prepare_medline_data():
    """
    Parse XML and save cleaned data to CSV.
    """
    # Paths
    project_root = Path(__file__).parent.parent
    xml_path = project_root / 'data' / 'medplus.xml'
    output_csv = project_root / 'data' / 'medline_cleaned.csv'
    
    print("📚 Parsing MedlinePlus XML...")
    df = parse_medlineplus_xml(xml_path)
    
    print(f"\n📊 Data Statistics:")
    print(f"   Total topics: {len(df)}")
    print(f"   Avg summary length: {df['summary'].str.len().mean():.0f} characters")
    print(f"   Topics with alt names: {(df['also_called'] != '').sum()}")
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved cleaned data to: {output_csv}")
    
    # Show sample
    print(f"\n📋 Sample entries:")
    for idx, row in df.head(3).iterrows():
        print(f"\n{idx+1}. {row['title']}")
        print(f"   Also called: {row['also_called'] or 'N/A'}")
        print(f"   Summary: {row['summary'][:150]}...")
    
    return df


if __name__ == "__main__":
    df = prepare_medline_data()