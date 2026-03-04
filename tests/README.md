# Tests

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_rag.py::TestRAGSystem::test_faq_accuracy -v -s
```

## Test Categories

### RAG System Tests
- `test_basic_query` - Basic query functionality
- `test_faq_accuracy` - FAQ accuracy from loaded documents (60% threshold)
- `test_conversation_memory` - Memory tracking across messages
- `test_long_conversation` - Handle 16+ message conversation
- `test_speed_performance` - Response time under 15s average
- `test_no_rag_fallback` - Works without RAG documents

### API Endpoint Tests
- `test_health_endpoint` - /health check
- `test_cms_endpoint` - /cms page loads
- `test_upload_endpoint` - File upload works
- `test_reload_endpoint` - Knowledge base reload works
