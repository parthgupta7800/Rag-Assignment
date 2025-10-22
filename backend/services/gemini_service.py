"""
Google Gemini API service for embeddings and chat completions.
"""
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    """Service class for Google Gemini API interactions."""
    
    def __init__(self):
        """Initialize the Gemini service with API key."""
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        logger.info(f"Initialized Gemini service with model: {settings.GEMINI_MODEL}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Gemini.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query text.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector
        """
        try:
            result = genai.embed_content(
                model=settings.EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            
            logger.info("Generated query embedding")
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response using Gemini with context and conversation history.
        
        Args:
            query: User's question
            context: Retrieved context from vector database
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response
        """
        try:
            # Build the prompt with context and history
            prompt = self._build_prompt(query, context, conversation_history)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            logger.info("Generated response using Gemini")
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _build_prompt(
        self, 
        query: str, 
        context: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a comprehensive prompt for the Gemini model.
        
        Args:
            query: User's question
            context: Retrieved context
            conversation_history: Previous messages
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # System instructions
        prompt_parts.append("""You are a helpful AI assistant specializing in electrical codes and company information. 
You can answer questions about:
1. NEC (National Electrical Code) guidelines and regulations
2. Wattmonk company information, policies, and services
3. General electrical and technical questions

Instructions:
- Use the provided context to answer questions accurately
- If the context doesn't contain relevant information, use your general knowledge
- Always cite your sources when using provided context
- Be concise but comprehensive in your responses
- If you're unsure about something, acknowledge the uncertainty""")
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("\nConversation History:")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        # Add context
        if context.strip():
            prompt_parts.append(f"\nRelevant Context:\n{context}")
        
        # Add current query
        prompt_parts.append(f"\nUser Question: {query}")
        prompt_parts.append("\nAssistant Response:")
        
        return "\n".join(prompt_parts)
    
    def classify_intent(self, query: str) -> str:
        """
        Classify the intent of a user query to determine the appropriate context.
        
        Args:
            query: User's question
            
        Returns:
            Intent classification: 'NEC', 'WATTMONK', or 'GENERAL'
        """
        try:
            classification_prompt = f"""
Classify the following query into one of these categories:
- NEC: Questions about electrical codes, regulations, safety standards, wiring, electrical installations, GFCI, grounding, circuits
- WATTMONK: Questions about Wattmonk company, services, policies, team, business operations, Zippy tool, solar engineering, plan sets, site surveys, permit applications
- GENERAL: General questions not specifically about NEC codes or Wattmonk

Important context:
- "Zippy" refers to Wattmonk's semi-automated plan set tool
- Questions about solar engineering tools, plan sets, site surveys are typically WATTMONK related

Query: "{query}"

Respond with only one word: NEC, WATTMONK, or GENERAL
"""
            
            response = self.model.generate_content(classification_prompt)
            intent = response.text.strip().upper()
            
            # Validate response
            if intent not in ['NEC', 'WATTMONK', 'GENERAL']:
                intent = 'GENERAL'
            
            logger.info(f"Classified query intent as: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            return 'GENERAL'

# Create global instance
gemini_service = GeminiService()
