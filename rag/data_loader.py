import xml.etree.ElementTree as ET
import html
import re
import pandas as pd
from pathlib import Path


def clean_html_text(html_text):
    """Remove HTML tags and clean text."""
    # Decode HTML entities (e.g., &lt; to <)
    text = html.unescape(html_text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_medlineplus_xml(xml_path):
    """
    Parse MedlinePlus XML and extract health topics.
    
    Returns:
        pd.DataFrame with columns: title, also_called, summary, url, id
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    topics = []
    
    for health_topic in root.findall('health-topic'):
        language = health_topic.get('language', '')
        if language != 'English':
            continue
        # Extract basic info
        title = health_topic.get('title', '')
        topic_id = health_topic.get('id', '')
        url = health_topic.get('url', '')
        
        # Extract alternative names
        also_called = [ac.text for ac in health_topic.findall('also-called') if ac.text]
        also_called_str = ', '.join(also_called) if also_called else ''
        
        # Extract and clean summary
        full_summary = health_topic.find('full-summary')
        if full_summary is not None and full_summary.text:
            summary = clean_html_text(full_summary.text)
        else:
            summary = ''
        
        # Skip if no meaningful content
        if not summary or len(summary) < 50:
            continue
        
        topics.append({
            'id': topic_id,
            'title': title,
            'also_called': also_called_str,
            'summary': summary,
            'url': url
        })
    
    df = pd.DataFrame(topics)
    print(f"✓ Parsed {len(df)} health topics from XML")
    return df


if __name__ == "__main__":
    # Test the parser
    xml_path = Path(__file__).parent.parent / 'data' / 'medplus.xml'
    df = parse_medlineplus_xml(xml_path)
    print(f"\nSample topics:\n{df[['title', 'summary']].head(3)}")
    print(f"\nDataFrame shape: {df.shape}")