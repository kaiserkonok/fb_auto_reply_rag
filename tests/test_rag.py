"""
Automated tests for RAG system and chatbot functionality.
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RAGSystem, init_memory_db


class TestRAGSystem:
    """Test RAG system functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        init_memory_db()
        self.rag = RAGSystem(
            upload_folder="uploads",
            llm_model="qwen2.5:3b",
            max_token_limit=2000
        )
        yield
        self.rag = None

    def test_basic_query(self):
        """Test basic query returns response."""
        result = self.rag.query("Hello, who are you?")
        assert isinstance(result, dict)
        assert "response" in result
        assert len(result["response"]) > 0

    def test_faq_accuracy(self):
        """Test FAQ accuracy - should answer from loaded documents."""
        faq_questions = [
            ("How do I start algorithmic trading?", "Sign up"),
            ("What is your pricing?", "199"),
            ("Who is the CEO?", "John Smith"),
            ("What markets do you support?", "stocks"),
            ("What is your cancellation policy", "30 days"),
        ]

        results = []
        for question, expected_keyword in faq_questions:
            result = self.rag.query(question)
            response = result.get("response", "").lower()
            found = expected_keyword.lower() in response
            results.append({
                "question": question,
                "expected": expected_keyword,
                "response": result.get("response", "")[:100],
                "found": found
            })
            print(f"\nQ: {question}")
            print(f"Expected: {expected_keyword}")
            print(f"Response: {result.get('response', '')[:100]}...")
            print(f"Found: {found}")

        accuracy = sum(1 for r in results if r["found"]) / len(results) * 100
        print(f"\n=== FAQ Accuracy: {accuracy:.1f}% ===")
        assert accuracy >= 60, f"FAQ accuracy {accuracy}% is below 60%"

    def test_conversation_memory(self):
        """Test conversation memory tracks context."""
        user_id = "test_user_memory"

        self.rag.query("My name is John", user_id=user_id)
        self.rag.query("I'm interested in trading", user_id=user_id)

        result = self.rag.query("What is my name?", user_id=user_id)
        response = result.get("response", "").lower()

        print(f"\nMemory test response: {response}")
        assert "john" in response or "name" in response, "Memory not working"

    def test_long_conversation(self):
        """Test handling of long conversation."""
        user_id = "test_long_conv"

        messages = [
            "Hi, I'm looking for trading solutions",
            "I have $50,000 to invest",
            "I prefer stocks and options",
            "What's your best plan?",
            "Do you offer API access?",
            "What's the setup process?",
            "How long does it take?",
            "Can I cancel anytime?",
            "What about support?",
            "Thanks, that's helpful",
            "One more question - do you teach beginners?",
            "What's the minimum experience needed?",
            "Okay, how do I sign up?",
            "Perfect, I'll do that",
            "Any special offers for new users?",
            "Great, talk soon!",
        ]

        start_time = time.time()
        for msg in messages:
            result = self.rag.query(msg, user_id=user_id)
            print(f"\nQ: {msg[:50]}...")
            print(f"A: {result.get('response', '')[:80]}...")

        total_time = time.time() - start_time
        avg_time = total_time / len(messages)

        print(f"\n=== Long Conversation Test ===")
        print(f"Total messages: {len(messages)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per message: {avg_time:.2f}s")

        assert total_time < 120, f"Long conversation too slow: {total_time:.2f}s"

    def test_speed_performance(self):
        """Test response speed."""
        queries = [
            "What services do you offer?",
            "How much does it cost?",
            "Who are you?",
        ]

        times = []
        for query in queries:
            start = time.time()
            result = self.rag.query(query)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"\nQuery: {query[:30]}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Response: {result.get('response', '')[:80]}...")

        avg_time = sum(times) / len(times)
        print(f"\n=== Speed Test ===")
        print(f"Average response time: {avg_time:.2f}s")

        assert avg_time < 15, f"Average response time {avg_time:.2f}s too slow"

    def test_no_rag_fallback(self):
        """Test fallback when no RAG documents."""
        rag_no_docs = RAGSystem(
            upload_folder="nonexistent_folder",
            llm_model="qwen2.5:3b",
            max_token_limit=2000
        )

        result = rag_no_docs.query("Hello!")
        assert isinstance(result, dict)
        assert "response" in result
        print(f"\nFallback response: {result.get('response', '')[:80]}")


class TestAPIEndpoints:
    """Test Flask API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, client):
        """Setup test client."""
        from wsgi import create_app
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_cms_endpoint(self):
        """Test CMS page loads."""
        response = self.client.get("/cms")
        assert response.status_code == 200
        assert b"Document Management" in response.data

    def test_upload_endpoint(self):
        """Test file upload."""
        from io import BytesIO
        data = {
            'file': (BytesIO(b"Test content"), 'test.txt')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200

    def test_reload_endpoint(self):
        """Test knowledge base reload."""
        response = self.client.post('/reload')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
