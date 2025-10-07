from typing import List, Dict


FOLLOW_UP_SYSTEM_PROMPT = """You are a medical assistant helping to gather information about a patient's symptoms. 
Your role is to ask ONE relevant follow-up question at a time to better understand their condition.

Guidelines:
- Ask questions that help narrow down the diagnosis
- Focus on severity, duration, associated symptoms, or risk factors
- Provide 3-4 clear multiple choice options
- Keep questions clear and concise
- Be empathetic and professional

CRITICAL: You MUST use this EXACT format with each option on a NEW LINE:

Question: [Your question here]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4 - if needed]

EACH OPTION MUST BE ON ITS OWN LINE. Do not put options on the same line."""

def create_followup_prompt(conversation_history: List[Dict[str, str]], question_num: int) -> str:
    """
    Create prompt for generating multiple choice follow-up questions.
    
    Args:
        conversation_history: List of {role, content} messages
        question_num: Which follow-up question number (1-4)
    
    Returns:
        Formatted prompt string
    """
    # Extract only user symptoms (not previous questions)
    user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    symptoms_summary = " | ".join(user_messages)
    
    # Get last question and answer if exists
    last_qa = ""
    if len(conversation_history) >= 2:
        last_q = conversation_history[-2]['content'] if conversation_history[-2]['role'] == 'assistant' else ""
        last_a = conversation_history[-1]['content'] if conversation_history[-1]['role'] == 'user' else ""
        if last_q and last_a:
            last_qa = f"\nLast Question Asked: {last_q}\nPatient's Answer: {last_a}\n"
    
    prompt = f"""{FOLLOW_UP_SYSTEM_PROMPT}

Patient's symptoms so far: {symptoms_summary}
{last_qa}
This is follow-up question #{question_num} of 4.

IMPORTANT: 
- Do NOT repeat questions already asked
- Build upon previous answers
- Focus on NEW aspects that help narrow down the diagnosis
- EACH OPTION MUST BE ON A SEPARATE LINE

Generate ONE NEW multiple choice follow-up question with 3-4 options.

You MUST use this EXACT format (note: each option on its own line):

Question: [Your question]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option - optional]

EXAMPLE of correct format:
Question: How severe is your fever?
A) Mild (under 100°F)
B) Moderate (100-102°F)
C) High (102-104°F)
D) Very high (above 104°F)

Now generate the question with EACH OPTION ON A NEW LINE:"""
    
    return prompt

DIAGNOSIS_SYSTEM_PROMPT = """You are a knowledgeable medical AI assistant. Based on the patient's symptoms and medical reference information, provide a comprehensive health assessment.

CRITICAL REQUIREMENTS:
1. Ground your response in the provided medical references
2. Include citations to the source materials
3. Provide a structured response with all required sections
4. Be empathetic but accurate
5. Include appropriate disclaimers about seeking professional care

OUTPUT STRUCTURE (must include all sections):
1. **Likely Condition**: What the patient might be experiencing and why
2. **30/60/90-Day Outlook**: Expected progression over time
3. **Lifestyle Recommendations**: Self-care and home management tips
4. **Red Flags & Next Steps**: Warning signs requiring immediate medical attention
5. **Citations**: References to source materials used"""


def create_diagnosis_prompt(
    conversation_history: List[Dict[str, str]], 
    retrieved_context: str
) -> str:
    """
    Create prompt for final diagnosis and recommendations.
    
    Args:
        conversation_history: Full conversation between user and assistant
        retrieved_context: Medical information from RAG retrieval
    
    Returns:
        Formatted prompt for diagnosis
    """
    # Format conversation
    history_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation_history
    ])
    
    prompt = f"""{DIAGNOSIS_SYSTEM_PROMPT}

PATIENT CONVERSATION:
{history_text}

MEDICAL REFERENCE INFORMATION:
{retrieved_context}

Based on the patient's symptoms described in the conversation and the medical reference information provided above, generate a comprehensive health assessment.

Format your response with these sections:

## Likely Condition
[Explain what the patient might be experiencing, linking their symptoms to the condition. Cite sources.]

## Expected Progression (30/60/90 Days)
- **30 days**: [What to expect in the first month]
- **60 days**: [What to expect after two months]
- **90 days**: [Long-term outlook]

## Lifestyle Recommendations
[Provide practical self-care advice, home remedies, and lifestyle modifications]

## Red Flags & Next Steps
[List warning signs that require immediate medical attention and recommended actions]

## Important Disclaimer
This AI assessment is for informational purposes only and is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.

Remember to cite the sources (e.g., "According to information about [Condition]...") and maintain a caring, professional tone."""
    
    return prompt


# JSON schema for structured output (optional - for future use)
DIAGNOSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {
            "type": "string",
            "description": "The likely medical condition"
        },
        "explanation": {
            "type": "string",
            "description": "Why this condition matches the symptoms"
        },
        "progression": {
            "type": "object",
            "properties": {
                "30_days": {"type": "string"},
                "60_days": {"type": "string"},
                "90_days": {"type": "string"}
            }
        },
        "lifestyle_recommendations": {
            "type": "array",
            "items": {"type": "string"}
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "citations": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["condition", "explanation", "progression", "lifestyle_recommendations", "red_flags"]
}


if __name__ == "__main__":
    # Test prompts
    print("="*60)
    print("FOLLOW-UP QUESTION PROMPT EXAMPLE")
    print("="*60)
    
    test_history = [
        {"role": "user", "content": "I have a fever and headache for 3 days"}
    ]
    
    followup = create_followup_prompt(test_history, 1)
    print(followup)
    
    print("\n" + "="*60)
    print("DIAGNOSIS PROMPT EXAMPLE")
    print("="*60)
    
    full_history = [
        {"role": "user", "content": "I have a fever and headache for 3 days"},
        {"role": "assistant", "content": "Do you have any body aches or chills?"},
        {"role": "user", "content": "Yes, I have body aches"},
        {"role": "assistant", "content": "Have you been in contact with anyone who was sick?"},
        {"role": "user", "content": "My child was sick last week"}
    ]
    
    test_context = "[Source: Influenza]\nInfluenza (flu) causes fever, body aches, headache..."
    
    diagnosis = create_diagnosis_prompt(full_history, test_context)
    print(diagnosis[:800] + "...")