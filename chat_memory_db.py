"""
Database for chat memory persistence.
Supports SQLite for local deployment.
"""

import sqlite3
import json
import logging
import os
import threading
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Thread-local storage for database connections
_local = threading.local()

# Database configuration
DB_PATH = os.getenv("CHAT_MEMORY_DB", "data/chat_memory.db")
DB_LOCK = threading.RLock()


def get_connection():
    """Get thread-local database connection."""
    if not hasattr(_local, 'connection'):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row
    return _local.connection


@contextmanager
def get_cursor():
    """Context manager for database cursor."""
    conn = get_connection()
    try:
        yield conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Initialize database tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_cursor() as cursor:
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            )
        ''')
        
        # Chat memory table - stores conversation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'human' or 'ai'
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_sessions(user_id)
            )
        ''')
        
        # Chat summary table - stores LLM-generated summaries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_summaries (
                user_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_sessions(user_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_memory_user_id 
            ON chat_memory(user_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_memory_created_at 
            ON chat_memory(created_at)
        ''')
        
    logger.info(f"Chat memory database initialized at {DB_PATH}")


def get_user_session(user_id):
    """Get or create user session."""
    with get_cursor() as cursor:
        cursor.execute(
            'SELECT * FROM user_sessions WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            cursor.execute(
                'INSERT INTO user_sessions (user_id) VALUES (?)',
                (user_id,)
            )
            cursor.execute(
                'SELECT * FROM user_sessions WHERE user_id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
        
        return dict(row) if row else None


def update_user_activity(user_id):
    """Update user's last active time and message count."""
    with get_cursor() as cursor:
        cursor.execute('''
            UPDATE user_sessions 
            SET last_active = CURRENT_TIMESTAMP, 
                message_count = message_count + 1 
            WHERE user_id = ?
        ''', (user_id,))


def save_message(user_id, role, content):
    """Save a chat message to database."""
    with get_cursor() as cursor:
        cursor.execute(
            'INSERT INTO chat_memory (user_id, role, content) VALUES (?, ?, ?)',
            (user_id, role, content)
        )
    
    update_user_activity(user_id)


def get_conversation_history(user_id, limit=20):
    """Get recent conversation history for a user."""
    with get_cursor() as cursor:
        cursor.execute('''
            SELECT role, content FROM chat_memory
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        # Return in chronological order
        return [dict(row) for row in reversed(rows)]


def save_summary(user_id, summary):
    """Save or update chat summary for a user."""
    with get_cursor() as cursor:
        cursor.execute('''
            INSERT OR REPLACE INTO chat_summaries (user_id, summary, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, summary))


def get_summary(user_id):
    """Get chat summary for a user."""
    with get_cursor() as cursor:
        cursor.execute(
            'SELECT summary FROM chat_summaries WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        return row['summary'] if row else None


def get_all_users():
    """Get all users (for admin purposes)."""
    with get_cursor() as cursor:
        cursor.execute('''
            SELECT user_id, created_at, last_active, message_count 
            FROM user_sessions 
            ORDER BY last_active DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]


def cleanup_old_messages(user_id, keep_last=50):
    """Cleanup old messages, keeping only the most recent ones."""
    with get_cursor() as cursor:
        # Get IDs of messages to keep
        cursor.execute('''
            SELECT id FROM chat_memory
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, keep_last))
        
        keep_ids = [row['id'] for row in cursor.fetchall()]
        
        if keep_ids:
            placeholders = ','.join('?' * len(keep_ids))
            cursor.execute(f'''
                DELETE FROM chat_memory
                WHERE user_id = ? AND id NOT IN ({placeholders})
            ''', (user_id, *keep_ids))


def delete_user(user_id):
    """Delete user and all their chat data."""
    with get_cursor() as cursor:
        cursor.execute('DELETE FROM chat_memory WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM chat_summaries WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))


def get_stats():
    """Get database statistics."""
    with get_cursor() as cursor:
        cursor.execute('SELECT COUNT(*) as count FROM user_sessions')
        users = cursor.fetchone()['count']
        
        cursor.execute('SELECT COUNT(*) as count FROM chat_memory')
        messages = cursor.fetchone()['count']
        
        cursor.execute('SELECT SUM(message_count) as total FROM user_sessions')
        total_messages = cursor.fetchone()['total'] or 0
        
        return {
            'total_users': users,
            'total_messages': messages,
            'total_conversations': total_messages
        }
