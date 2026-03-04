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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s \033[96m%(name)s\033[0m \033[93m%(levelname)s\033[0m - %(message)s",
)
logger = logging.getLogger(__name__)


def init_memory_db():
    pass


class RAGSystem:
    def __init__(self, upload_folder='uploads', llm_model="qwen2.5:3b", embed_model="bge-m3:latest", max_token_limit=2000):
        self.upload_folder = upload_folder
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vector_store = None
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        self.max_token_limit = max_token_limit
        self.user_memories = {}
        logger.info(f"Initializing RAG system with {llm_model} (max_token_limit={max_token_limit})...")
        self.load_documents()

    def _get_memory(self, user_id):
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
        return self.user_memories[user_id]
    
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

            # Retrieval - using MMR for better diversity
            context_section = ""
            retrieved_docs = []
            if self.vector_store:
                logger.debug(f"\n🔍 SEARCHING for: '{message}'")
                
                # Use MMR (Maximum Marginal Relevance) for better results
                docs = self.vector_store.max_marginal_relevance_search(
                    message, 
                    k=5,  # Increased from 3 to 5
                    fetch_k=10  # Fetch more for diversity
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

            # Build improved prompt
            prompt = f"""You are a helpful assistant for "Algo Trade Pro" - an algorithmic trading company.

IMPORTANT: You must ONLY use the information provided in the "Relevant information" section below to answer questions. 
If the information is not in the documents, say "I don't have that information" - do NOT make up answers.

Conversation summary from earlier:
{summary_text}

Recent conversation:
{history_text}

{context_section}

Based ONLY on the information above, please answer this question from the user:

User's question: {message}

Your answer (use only information from the documents):"""

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

            memory.save_context({"input": message}, {"output": answer})

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
