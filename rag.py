"""
RAG Engine - Core retrieval and question answering system.
"""

import os
import logging
import json
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Please see the migration guide.*")

# Suppress ChromaDB telemetry warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)

from langchain.memory import ConversationSummaryBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chat_memory_db import (
    init_db,
    get_user_session,
    save_message,
    get_conversation_history,
    save_summary,
    get_summary,
    cleanup_old_messages,
    get_stats
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s \033[96m%(name)s\033[0m \033[93m%(levelname)s\033[0m - %(message)s",
)
logger = logging.getLogger(__name__)


def init_memory_db():
    """Initialize the chat memory database."""
    init_db()


class RAGSystem:
    def __init__(self, upload_folder='uploads', llm_model="qwen2.5:3b", embed_model="embeddinggemma", max_token_limit=2000):
        self.upload_folder = upload_folder
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vector_store = None
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        self.max_token_limit = max_token_limit
        self.user_memories = {}
        self.user_memory_loaded = {}  # Track which users have memory loaded from DB
        logger.info(f"Initializing RAG system with {llm_model} (max_token_limit={max_token_limit})...")
        
        # Initialize database
        init_db()
        
        # Log stats
        stats = get_stats()
        logger.info(f"Chat DB stats: {stats['total_users']} users, {stats['total_messages']} messages")
        
        self.load_documents()

    def _get_memory(self, user_id):
        """Get or create memory for a user, loading from DB if needed."""
        if user_id not in self.user_memories:
            # Create new memory
            self.user_memories[user_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
            
            # Load existing conversation from database
            if user_id not in self.user_memory_loaded:
                self._load_user_memory_from_db(user_id)
                self.user_memory_loaded[user_id] = True
        
        return self.user_memories[user_id]
    
    def _load_user_memory_from_db(self, user_id):
        """Load user's conversation history from database."""
        try:
            # Load summary
            summary = get_summary(user_id)
            if summary:
                # Set the summary in memory
                self.user_memories[user_id].buffer = summary
                logger.debug(f"Loaded summary for user {user_id}")
            
            # Load recent messages
            history = get_conversation_history(user_id, limit=20)
            
            for msg in history:
                if msg['role'] == 'human':
                    self.user_memories[user_id].chat_memory.add_user_message(msg['content'])
                elif msg['role'] == 'ai':
                    self.user_memories[user_id].chat_memory.add_ai_message(msg['content'])
            
            logger.debug(f"Loaded {len(history)} messages for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error loading memory from DB for user {user_id}: {e}")
    
    def _save_message_to_db(self, user_id, role, content):
        """Save message to database."""
        try:
            save_message(user_id, role, content)
            
            # Periodic cleanup - keep last 100 messages per user
            cleanup_old_messages(user_id, keep_last=100)
            
        except Exception as e:
            logger.error(f"Error saving message to DB: {e}")
    
    def _save_summary_to_db(self, user_id):
        """Save conversation summary to database."""
        try:
            memory = self.user_memories.get(user_id)
            if memory:
                # Get current summary from memory
                memory_variables = memory.load_memory_variables({})
                summary = memory_variables.get("summary", "")
                if summary:
                    save_summary(user_id, summary)
                    logger.debug(f"Saved summary for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving summary to DB: {e}")
    
    def reload(self):
        logger.info("Reloading knowledge base...")
        self.load_documents()
        logger.info("Knowledge base reloaded")
    
    def query(self, message, user_id=None):
        """
        Query the RAG system.
        
        Args:
            message: User message
            user_id: If provided, uses ConversationSummaryBufferMemory for conversation history.
                     If None, uses fresh memory (no persistence).
        """
        if not message:
            return {"error": "No message provided"}, 400

        if not user_id:
            user_id = "web"

        try:
            memory = self._get_memory(user_id)
            memory_variables = memory.load_memory_variables({})
            history_text = memory_variables.get("history", "")
            summary_text = memory_variables.get("summary", "")

            logger.debug(f"\n{'='*50}")
            logger.debug(f"USER: {message}")
            logger.debug(f"{'='*50}")

            # Check if message needs RAG (not a simple greeting/casual message)
            message_lower = message.lower().strip()
            simple_queries = ['hi', 'hello', 'hey', 'ok', 'thanks', 'thank you', 'good', 'nice', 'cool', 'okay', 'sure', 'yes', 'no']
            needs_rag = not any(message_lower == q or message_lower.startswith(q + ' ') for q in simple_queries)

            # Retrieval - using MMR for better diversity (only if needed)
            context_section = ""
            retrieved_docs = []
            if self.vector_store and needs_rag:
                logger.debug(f"\n🔍 SEARCHING for: '{message}'")
                
                # Use MMR (Maximum Marginal Relevance) for better results
                docs = self.vector_store.max_marginal_relevance_search(
                    message, 
                    k=5,
                    fetch_k=10
                )
                logger.debug(f"📄 FOUND {len(docs)} documents")
                
                for i, doc in enumerate(docs):
                    logger.debug(f"\n--- Document {i+1} ---")
                    logger.debug(doc.page_content[:500])
                    retrieved_docs.append(doc.page_content)
                
                if docs:
                    context_section = "\n\nRelevant information from documents:\n" + "\n".join(
                        f"[Document {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)
                    )
            else:
                logger.debug(f"⏭️ SKIPPING RAG for simple message")

            # Build improved prompt
            prompt = f"""You are a helpful assistant for "Algo Trade Pro" - an algorithmic trading company.

For greetings and casual conversation (like "hi", "hello", "how are you"), respond naturally and friendly - no need for documents.

For questions about the company, services, pricing, or trading: Use the "Relevant information" section below if available. If the information is not in the documents, use your general knowledge about algorithmic trading to help - do NOT say "I don't have that information".

Conversation summary from earlier:
{summary_text}

Recent conversation:
{history_text}

{context_section}

User's question: {message}

Your friendly answer:"""

            logger.debug(f"\n{'='*50}")
            logger.debug(f"CONTEXT USED:")
            logger.debug(f"Summary: {summary_text[:300] if summary_text else 'None'}")
            logger.debug(f"History: {history_text[:300] if history_text else 'None'}")
            logger.debug(f"Retrieved: {len(retrieved_docs)} docs")
            logger.debug(f"{'='*50}")

            # Generate response
            logger.debug(f"\n🤖 GENERATING response...")
            answer = str(self.llm.invoke(prompt)).strip()
            if not answer:
                answer = "I am here. Tell me what you want to talk about."

            logger.debug(f"\n✅ RESPONSE: {answer[:200]}...")
            logger.debug(f"{'='*50}\n")

            # Update in-memory memory
            memory.save_context({"input": message}, {"output": answer})
            
            # Save to database for persistence
            self._save_message_to_db(user_id, "human", message)
            self._save_message_to_db(user_id, "ai", answer)
            
            # Save summary periodically (every 10 messages)
            try:
                session = get_user_session(user_id)
                if session and session.get('message_count', 0) % 10 == 0:
                    self._save_summary_to_db(user_id)
            except:
                pass

            logger.info(f"Query for user {user_id}: {message[:30]}...")
            return {"response": answer}

        except Exception as e:
            logger.error(f"Query error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"response": f"Error processing your query: {str(e)}"}, 500
    
    def load_documents(self):
        # Clear existing vector store first
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except:
                pass
            self.vector_store = None
        
        if not os.path.exists(self.upload_folder):
            return []
        
        documents = []
        for root, dirs, files in os.walk(self.upload_folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                try:
                    if filename.endswith('.txt'):
                        loader = TextLoader(filepath, encoding='utf-8')
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                    elif filename.endswith('.pdf'):
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                    elif filename.endswith('.docx'):
                        loader = Docx2txtLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                    elif filename.endswith('.csv'):
                        loader = CSVLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        if documents:
            # Improved chunking for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,      # Smaller chunks for more precise retrieval
                chunk_overlap=100,    # Less overlap
                separators=["\n\n", "\n", ". ", ", ", " "]
            )
            
            docs = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(docs)} chunks")
            
            self.vector_store = Chroma.from_documents(
                docs, 
                self.embeddings,
                collection_name="rag-store"
            )
            logger.info(f"Vector store created with {len(docs)} documents")
        
        return documents
