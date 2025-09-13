# Greeting and Intent Detection Functions

import re

_DOC_KEYWORDS = ['phase', 'processus', 'projet', 'document', 'étape', 'développement', 'phases', 'documents']

def detect_intent(user_input):
    """Detect user intent to provide appropriate responses"""
    
    user_input_lower = user_input.lower().strip()
    # Normalize punctuation/spaces so patterns like "hi, how are you" are caught
    norm = re.sub(r"[\.,!?:;]+", " ", user_input_lower)
    norm = re.sub(r"\s+", " ", norm).strip()
    
    # Simple greeting patterns
    greeting_patterns = [
        r'^(hello|hi|hey|bonjour|salut)(\s.*)?$',
        r'^(how are you|comment allez[- ]vous|ça va|comment ça va)(\s.*)?$',
        r'^(good (morning|afternoon|evening)|bonsoir|bonne journée)(\s.*)?$',
        r'^(merci|thank you|thanks)(\s.*)?$',
        r'^(au revoir|goodbye|bye)(\s.*)?$'
    ]
    
    # Simple chat patterns (not greetings but basic conversation)
    simple_chat_patterns = [
        r'^(ça va|comment ça va|how are you doing)(\s.*)?$',
        r'^(quoi de neuf|what\'s up|quoi de nouveau)(\s.*)?$',
        r'^(comment vas-tu|how do you do)(\s.*)?$'
    ]
    
    # Document analysis patterns
    document_patterns = [
        r'\b(phase|étape|processus|projet|développement)\b',
        r'\b(combien|how many|liste|list)\b',
        r'\b(que|what|pourquoi|why|comment|how)\b.*\b(phase|document|processus)\b',
        r'\b(décris|describe|explique|explain)\b'
    ]
    
    # If message has greeting and no document keywords, treat as greeting
    if not any(k in norm for k in _DOC_KEYWORDS):
        for pattern in greeting_patterns:
            if re.search(pattern, norm):
                return "greeting"
        for pattern in simple_chat_patterns:
            if re.search(pattern, norm):
                return "simple_chat"

    # Check for greetings (fallback even if mixed)
    for pattern in greeting_patterns:
        if re.search(pattern, norm):
            return "greeting"
    
    # Check for simple chat
    for pattern in simple_chat_patterns:
        if re.search(pattern, norm):
            return "simple_chat"
    
    # Check for document analysis requests
    for pattern in document_patterns:
        if re.search(pattern, norm):
            return "document_analysis"
    
    # If input is very short and doesn't contain document keywords
    if len(norm.split()) <= 3 and not any(keyword in norm for keyword in _DOC_KEYWORDS):
        return "simple_chat"
    
    # Default to document analysis for longer queries
    # Prefer simple chat if the sentence is composed mostly of greeting words
    greeting_vocab = set(['hello','hi','hey','bonjour','salut','merci','thank','thanks','bye','goodbye','au','revoir','how','are','you','ça','va','comment','allez','vous','morning','afternoon','evening'])
    tokens = norm.split()
    if tokens:
        frac_greet = sum(1 for t in tokens if t in greeting_vocab) / len(tokens)
        if frac_greet >= 0.6 and not any(k in norm for k in _DOC_KEYWORDS):
            return "simple_chat"

    return "document_analysis"

def generate_greeting_response(lang: str = 'fr'):
    """Generate localized greeting; default French to preserve previous behavior."""
    import random
    fr = [
        "Bonjour ! Je suis votre assistant pour l'analyse de documents PDF. Comment puis-je vous aider aujourd'hui ?",
        "Salut ! Je suis là pour répondre à vos questions sur les documents que vous avez uploadés. Que souhaitez-vous savoir ?",
        "Bonjour ! Je peux vous aider à analyser et comprendre vos documents PDF. Posez-moi une question !",
        "Bonjour ! Je vais bien, merci ! Je suis votre assistant d'analyse documentaire. Comment puis-je vous aider ?"
    ]
    en = [
        "Hello! I'm your assistant for analyzing PDF documents. How can I help you today?",
        "Hi! I'm here to answer your questions about the documents you've uploaded. What would you like to know?",
        "Hello! I can help you analyze and understand your PDF documents. Ask me a question!",
        "Hi! I'm ready to analyze your documents. How can I assist?"
    ]
    return random.choice(fr if lang == 'fr' else en)

def generate_simple_chat_response(user_input, lang: str = 'fr'):
    """Generate simple chat responses localized to user language."""
    user_lower = user_input.lower()
    if lang == 'fr':
        if any(word in user_lower for word in ['merci']):
            return "De rien ! N'hésitez pas si vous avez d'autres questions sur vos documents."
        if any(word in user_lower for word in ['bye', 'au revoir']):
            return "Au revoir ! J'espère avoir pu vous aider avec l'analyse de vos documents."
        if any(word in user_lower for word in ['ça va', 'comment allez']):
            return "Je vais très bien, merci ! Je suis prêt à analyser vos documents PDF. Avez-vous des questions ?"
        return "Je suis votre assistant d'analyse de documents PDF. Posez-moi une question sur vos documents uploadés !"
    else:
        if any(word in user_lower for word in ['thank', 'thanks']):
            return "You're welcome! Feel free to ask anything else about your documents."
        if any(word in user_lower for word in ['bye', 'goodbye']):
            return "Goodbye! I hope I was helpful with your document analysis."
        if any(phrase in user_lower for phrase in ['how are you']):
            return "I'm doing great—ready to analyze your documents. Do you have a question?"
        return "I'm your PDF document analysis assistant. Ask me something about the documents you've uploaded!"

def should_use_enhanced_retrieval(intent, user_input):
    """Determine if enhanced retrieval should be used"""
    
    if intent in ["greeting", "simple_chat"]:
        return False
    
    # Use enhanced retrieval for document analysis
    if intent == "document_analysis":
        return True
    
    # Check for specific keywords that indicate need for document search
    document_keywords = ['phase', 'processus', 'projet', 'étape', 'développement', 'document', 'combien', 'liste']
    return any(keyword in user_input.lower() for keyword in document_keywords)
