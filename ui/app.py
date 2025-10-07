import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.rag_pipeline import RAGPipeline, ConversationManager


# Page config
st.set_page_config(
    page_title="Medical Symptom Checker",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached to avoid reloading)."""
    project_root = Path(__file__).parent.parent
    store_dir = project_root / 'store'
    return RAGPipeline(store_dir)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_manager' not in st.session_state:
        pipeline = load_pipeline()
        st.session_state.conversation_manager = ConversationManager(pipeline)
    if 'diagnosis_complete' not in st.session_state:
        st.session_state.diagnosis_complete = False


def display_sources(sources):
    """Display source citations."""
    st.markdown("---")
    st.markdown("### 📚 Sources")
    for i, source in enumerate(sources, 1):
        st.markdown(f"{i}. [{source['title']}]({source['url']}) (Relevance: {source['relevance_score']:.3f})")


def main():
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🏥 Medical Symptom Checker")
    st.markdown("*AI-powered preliminary health assessment based on MedlinePlus*")
    st.markdown("---")
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer - Please Read"):
        st.warning("""
        This AI chatbot is for **informational purposes only** and is not a substitute for professional medical advice, diagnosis, or treatment.
        
        - Always seek the advice of a qualified healthcare provider
        - Never disregard professional medical advice or delay seeking it
        - If you have a medical emergency, call emergency services immediately
        - This tool uses AI and may make mistakes
        """)
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show diagnosis if complete
    if st.session_state.diagnosis_complete:
        st.success("✅ Assessment Complete!")
        
        # Display sources if available
        if 'sources' in st.session_state:
            display_sources(st.session_state.sources)
        
        if st.button("🔄 Start New Assessment"):
            # Reset everything
            st.session_state.messages = []
            pipeline = load_pipeline()
            st.session_state.conversation_manager = ConversationManager(pipeline)
            st.session_state.diagnosis_complete = False
            st.rerun()
        return
    
    # Initial symptom input
    if len(st.session_state.messages) == 0:
        st.markdown("### 👋 Welcome! Please describe your symptoms:")
        
        user_input = st.text_area(
            "Symptoms",
            placeholder="Example: I have a fever, headache, and body aches for 3 days...",
            height=120,
            label_visibility="collapsed",
            key="symptom_input"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.button("🔍 Get Diagnosis", type="primary", use_container_width=True)
        
        if submit_button:
            if user_input.strip():
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Show user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Process and get diagnosis
                with st.spinner("🔍 Analyzing your symptoms and retrieving medical information..."):
                    response = st.session_state.conversation_manager.process_message(user_input)
                
                # Add assistant diagnosis
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['content']
                })
                
                # Show diagnosis
                with st.chat_message("assistant"):
                    st.markdown(response['content'])
                
                st.session_state.diagnosis_complete = True
                st.session_state.sources = response.get('sources', [])
                st.rerun()
            else:
                st.error("⚠️ Please describe your symptoms before submitting.")


if __name__ == "__main__":
    main()