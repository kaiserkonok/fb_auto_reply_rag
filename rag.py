"""
RAG Engine - Core retrieval and question answering system.
"""

import os
import json
import warnings
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.rule import Rule
    from rich import box as box_module
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using plain print")

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def print_colored(title: str, content: str = "", style: str = "cyan"):
    """Print a panel using Rich or fallback to print."""
    if console:
        console.print(Panel(
            content or title,
            title=f"[bold {style}]{title}[/bold {style}]" if title else None,
            border_style=style,
            expand=False,
            padding=(1, 2)
        ))
    else:
        print(f"=== {title} ===")
        if content:
            print(content)


def print_rule(style: str = "cyan"):
    """Print a horizontal rule."""
    if console:
        console.print(Rule(style=style))
    else:
        print("-" * 60)


def print_step(step: str, detail: str = "", status: str = "pending"):
    """Print a step with status indicator."""
    status_icon = {"pending": "⏳", "running": "⚙️", "success": "✅", "error": "❌"}.get(status, "•")
    color = {"pending": "yellow", "running": "cyan", "success": "green", "error": "red"}.get(status, "white")
    
    if console:
        msg = f"[bold {color}]{status_icon} {step}[/bold {color}]"
        if detail:
            msg += f"\n   {detail}"
        console.print(msg)
    else:
        print(f"{status_icon} {step}")
        if detail:
            print(f"   {detail}")


def print_documents(docs: list, query: str = ""):
    """Print retrieved documents in a beautiful format."""
    if not docs:
        return
    
    if console:
        if query:
            console.print(f"\n[bold cyan]🔍 Search Query:[/bold cyan] [yellow]{query}[/yellow]\n")
        
        print_rule("cyan")
        console.print(f"[bold cyan]📚 Retrieved {len(docs)} Documents[/bold cyan]")
        print_rule("cyan")
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            content = doc.page_content[:400]
            if len(doc.page_content) > 400:
                content += "..."
            
            console.print(Panel(
                f"[dim]Source: {source}[/dim]\n\n{content}",
                title=f"[bold magenta]Document {i}[/bold magenta]",
                border_style="magenta",
                expand=False,
                padding=(1, 1)
            ))
            console.print()
    else:
        print(f"\n=== Retrieved {len(docs)} Documents ===")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            content = doc.page_content[:300]
            print(f"\n--- Document {i} ---")
            print(f"Source: {source}")
            print(f"Content: {content}...")


def print_prompt(prompt: str):
    """Print the LLM prompt in syntax-highlighted format."""
    if not console:
        print("\n=== LLM Prompt ===")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        return
    
    console.print()
    print_rule("yellow")
    console.print("[bold yellow]📝 LLM Prompt:[/bold yellow]")
    print_rule("yellow")
    
    display_prompt = prompt[:1500] + "..." if len(prompt) > 1500 else prompt
    syntax = Syntax(display_prompt, "markdown", theme="monokai", line_numbers=False)
    console.print(syntax)


def print_response(response: str):
    """Print the LLM response."""
    if not console:
        print("\n=== LLM Response ===")
        print(response[:300] + "..." if len(response) > 300 else response)
        return
    
    console.print()
    print_rule("green")
    console.print("[bold green]💬 LLM Response:[/bold green]")
    print_rule("green")
    
    display_response = response[:800] + "..." if len(response) > 800 else response
    syntax = Syntax(display_response, "markdown", theme="monokai", line_numbers=False)
    console.print(syntax)


def print_user(message: str, user_id: str = ""):
    """Print user message with styling."""
    if not console:
        print(f"\n=== User Message (@{user_id}) ===")
        print(message)
        return
    
    user_tag = f"[dim]@[/dim][yellow]{user_id}[/yellow]" if user_id else ""
    console.print()
    print_rule("orange3")
    console.print(f"[bold orange3]👤 User Message:[/bold orange3] {user_tag}")
    print_rule("orange3")
    console.print(f"[white]{message}[/white]")


def print_context(history: any, summary: str, docs_count: int):
    """Print context information."""
    # Handle history being a list or string
    if isinstance(history, list):
        history_str = str(history) if history else "None"
    else:
        history_str = str(history) if history else "None"
    
    summary_str = str(summary) if summary else "None"
    history_preview = history_str[:100] + "..." if len(history_str) > 100 else history_str
    summary_preview = summary_str[:100] + "..." if len(summary_str) > 100 else summary_str
    
    if not console:
        print("\n=== Context ===")
        print(f"History: {history_preview}")
        print(f"Summary: {summary_preview}")
        print(f"Documents: {docs_count}")
        return
    
    table = Table(title="[bold cyan]📊 Context[/bold cyan]", box=box_module.ROUNDED)
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="white")
    table.add_row("History", history_preview)
    table.add_row("Summary", summary_preview)
    table.add_row("Documents", str(docs_count))
    console.print(table)


def print_files(loaded_files: list):
    """Print loaded files table."""
    if not loaded_files:
        return
    
    if not console:
        print("\n=== Loaded Files ===")
        for f in loaded_files:
            print(f"  - {f}")
        return
    
    table = Table(title="[bold green]✅ Files Loaded[/bold green]", box=box_module.HEAVY)
    table.add_column("Filename", style="cyan")
    for f in loaded_files:
        table.add_row(f)
    console.print(table)


from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

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

warnings.filterwarnings("ignore")


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
        self.user_memory_loaded = {}
        
        print_colored(
            "🚀 RAG System Ready",
            f"LLM: {llm_model}\nEmbed: {embed_model}\nMax tokens: {max_token_limit}\nFolder: {upload_folder}",
            "cyan"
        )
        
        init_db()
        stats = get_stats()
        print(f"📊 Database: {stats['total_users']} users, {stats['total_messages']} messages")
        
        self.load_documents()

    def _get_memory(self, user_id):
        """Get or create memory for a user, loading from DB if needed."""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                return_messages=True
            )
            
            if user_id not in self.user_memory_loaded:
                self._load_user_memory_from_db(user_id)
                self.user_memory_loaded[user_id] = True
        
        return self.user_memories[user_id]
    
    def _load_user_memory_from_db(self, user_id):
        """Load user's conversation history from database."""
        try:
            summary = get_summary(user_id)
            if summary:
                self.user_memories[user_id].buffer = summary
            
            history = get_conversation_history(user_id, limit=20)
            
            for msg in history:
                if msg['role'] == 'human':
                    self.user_memories[user_id].chat_memory.add_user_message(msg['content'])
                elif msg['role'] == 'ai':
                    self.user_memories[user_id].chat_memory.add_ai_message(msg['content'])
            
        except Exception as e:
            print(f"⚠️ Error loading memory: {e}")
    
    def _save_message_to_db(self, user_id, role, content):
        """Save message to database."""
        try:
            save_message(user_id, role, content)
            cleanup_old_messages(user_id, keep_last=100)
        except Exception as e:
            print(f"⚠️ Error saving message: {e}")
    
    def _save_summary_to_db(self, user_id):
        """Save conversation summary to database."""
        try:
            memory = self.user_memories.get(user_id)
            if memory:
                memory_variables = memory.load_memory_variables({})
                summary = memory_variables.get("summary", "")
                if summary:
                    save_summary(user_id, summary)
        except Exception as e:
            print(f"⚠️ Error saving summary: {e}")
    
    def reload(self):
        print("♻️  Reloading knowledge base...")
        self.load_documents()
        print("✅ Knowledge base reloaded!")
    
    def _format_history(self, history):
        """Convert LangChain message objects to readable text format."""
        if not history:
            return "No previous conversation"
        
        lines = []
        for msg in history:
            # Extract content from LangChain message objects
            if hasattr(msg, 'content'):
                content = msg.content
                # Determine if human or AI
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        lines.append(f"Human: {content}")
                    elif msg.type == 'ai':
                        lines.append(f"AI: {content}")
                    else:
                        lines.append(f"{msg.type}: {content}")
                else:
                    lines.append(content)
            else:
                lines.append(str(msg))
        
        return "\n".join(lines)
    
    def query(self, message, user_id=None):
        """Query the RAG system."""
        if not message:
            return {"error": "No message provided"}, 400

        if not user_id:
            user_id = "web"

        try:
            memory = self._get_memory(user_id)
            memory_variables = memory.load_memory_variables({})
            
            # Convert LangChain message objects to readable text
            history_raw = memory_variables.get("history", [])
            history_text = self._format_history(history_raw)
            
            summary_text = memory_variables.get("summary", "")

            print_user(message, user_id)

            context_section = ""
            retrieved_docs = []
            if self.vector_store:
                print_step("Searching vector store", f'Query: "{message}"', "running")
                docs = self.vector_store.similarity_search(message, k=5)
                retrieved_docs = docs
                
                print_step(f"Found {len(docs)} documents", "", "success")
                print_documents(docs, message)
                
                if docs:
                    context_section = "\n\n📚 Relevant information:\n" + "\n".join(
                        f"• {doc.page_content[:200]}" for doc in docs
                    )

            prompt = f"""You are a friendly assistant for Algo Trade Pro, an algorithmic trading company.

You help users with questions about trading, services, pricing, and company info.

If the user greets you (hi, hello, hey), just respond naturally and warmly.

When you know the answer, give it confidently - don't add disclaimers like "based on the documents" or "in the information provided".

If you truly don't know something, say so simply.

Recent conversation:
{history_text}

{context_section}

User: {message}

Your response:"""

            print_context(history_text, summary_text, len(retrieved_docs))
            print_prompt(prompt)

            print_step("Generating response", f"Model: {self.llm.model}", "running")
            
            answer = str(self.llm.invoke(prompt)).strip()
            if not answer:
                answer = "I am here. Tell me what you want to talk about."

            print_response(answer)
            print_step("Response ready", f"Length: {len(answer)} chars", "success")

            memory.save_context({"input": message}, {"output": answer})
            
            # Only save to DB for real users (not test/homepage users)
            is_test_user = user_id in ("homepage-user", "web") or user_id.startswith("test")
            if not is_test_user:
                self._save_message_to_db(user_id, "human", message)
                self._save_message_to_db(user_id, "ai", answer)
                
                try:
                    session = get_user_session(user_id)
                    if session and session.get('message_count', 0) % 10 == 0:
                        self._save_summary_to_db(user_id)
                except:
                    pass

            print_rule("green")
            print(f"✅ Query complete for {user_id}\n")
            return {"response": answer}

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"Error processing your query: {str(e)}"}, 500
    
    def load_documents(self):
        self.vector_store = None
        
        if not os.path.exists(self.upload_folder):
            print(f"⚠️  Folder not found: {self.upload_folder}")
            return []
        
        documents = []
        loaded_files = []
        
        print(f"\n📂 Loading documents from {self.upload_folder}...")
        
        for root, dirs, files in os.walk(self.upload_folder):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                try:
                    if filename.endswith('.txt'):
                        loader = TextLoader(filepath, encoding='utf-8')
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_files.append(filename)
                    elif filename.endswith('.pdf'):
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_files.append(filename)
                    elif filename.endswith('.docx'):
                        loader = Docx2txtLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_files.append(filename)
                    elif filename.endswith('.csv'):
                        loader = CSVLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        loaded_files.append(filename)
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        print_files(loaded_files)
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", ", ", " "]
            )
            
            docs = text_splitter.split_documents(documents)
            print(f"✂️  Split into {len(docs)} chunks")
            
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            
            print_colored(
                "✅ Knowledge Base Ready",
                f"{len(documents)} documents\n{len(docs)} chunks\nFAISS vector store",
                "green"
            )
        
        return documents
