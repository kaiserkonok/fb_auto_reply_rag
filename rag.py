"""
RAG Engine - Core retrieval and question answering system.
"""

import os
import logging

from langchain.memory import ConversationSummaryBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

            context_section = ""
            if self.vector_store:
                docs = self.vector_store.similarity_search(message, k=3)
                if docs:
                    context_section = "\n\nRelevant information:\n" + "\n".join(
                        f"- {doc.page_content[:200]}" for doc in docs
                    )

            prompt = f"""You are a helpful, natural conversational assistant.
Respond like a human assistant in clear, friendly English.

Conversation summary:
{summary_text}

Recent conversation:
{history_text}
{context_section}

Human: {message}
Assistant:"""

            answer = str(self.llm.invoke(prompt)).strip()
            if not answer:
                answer = "I am here. Tell me what you want to talk about."

            memory.save_context({"input": message}, {"output": answer})

            logger.info(f"Query for user {user_id}: {message[:30]}...")
            return {"response": answer}

        except Exception as e:
            logger.error(f"Query error: {e}")
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
                        loader = TextLoader(filepath)
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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            docs = text_splitter.split_documents(documents)
            self.vector_store = Chroma.from_documents(docs, self.embeddings)
        
        return documents
