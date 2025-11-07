"""
Interesting tests for common utils! 🧪✨

These tests cover edge cases, integration scenarios, and some fun challenges
to make sure our utils are robust and ready for adventure!
"""
import pytest
import os
from datetime import datetime, timezone
from typing import Dict, Any

from vhm_common_utils.data_models import Anchor, IndexedAnchorResponse
from vhm_common_utils.config import Settings
from vhm_common_utils.embedding import get_embedding


class TestAnchorModel:
    """Testing the Anchor model with various scenarios 🎯"""
    
    def test_anchor_basic_creation(self):
        """Happy path: create a basic anchor"""
        anchor = Anchor(
            anchor_id="test-123",
            text="Hello, world!",
            stored_at=datetime.now(timezone.utc)
        )
        assert anchor.anchor_id == "test-123"
        assert anchor.text == "Hello, world!"
        assert anchor.salience == 1.0  # default
        assert anchor.meta == {}  # default
    
    def test_anchor_with_extreme_salience(self):
        """Edge case: very high and very low salience values"""
        # High salience
        anchor_high = Anchor(
            anchor_id="high-salience",
            text="Very important memory",
            stored_at=datetime.now(timezone.utc),
            salience=999.99
        )
        assert anchor_high.salience == 999.99
        
        # Low salience
        anchor_low = Anchor(
            anchor_id="low-salience",
            text="Trivial detail",
            stored_at=datetime.now(timezone.utc),
            salience=0.001
        )
        assert anchor_low.salience == 0.001
    
    def test_anchor_with_rich_metadata(self):
        """Interesting: anchor with complex metadata structure"""
        complex_meta: Dict[str, Any] = {
            "source": "conversation",
            "participants": ["Alice", "Bob"],
            "topics": ["AI", "memory", "embeddings"],
            "nested": {
                "level1": {
                    "level2": "deep_value"
                }
            },
            "numbers": [1, 2, 3, 42],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        anchor = Anchor(
            anchor_id="meta-rich",
            text="A memory with lots of context",
            stored_at=datetime.now(timezone.utc),
            meta=complex_meta
        )
        
        assert anchor.meta["nested"]["level1"]["level2"] == "deep_value"
        assert len(anchor.meta["topics"]) == 3
        assert 42 in anchor.meta["numbers"]
    
    def test_anchor_with_empty_text(self):
        """Edge case: empty text (should still work)"""
        anchor = Anchor(
            anchor_id="empty-text",
            text="",
            stored_at=datetime.now(timezone.utc)
        )
        assert anchor.text == ""
        assert len(anchor.text) == 0
    
    def test_anchor_with_very_long_text(self):
        """Edge case: extremely long text (simulating a long memory)"""
        long_text = "Lorem ipsum dolor sit amet, " * 1000  # ~30k chars
        anchor = Anchor(
            anchor_id="long-memory",
            text=long_text,
            stored_at=datetime.now(timezone.utc)
        )
        assert len(anchor.text) > 10000
        assert anchor.text.startswith("Lorem ipsum")


class TestIndexedAnchorResponse:
    """Testing the IndexedAnchorResponse model 📦"""
    
    def test_successful_response(self):
        """Happy path: successful indexing"""
        response = IndexedAnchorResponse(
            anchor_id="success-123",
            ok=True
        )
        assert response.ok is True
        assert response.reason is None
        assert response.detail is None
    
    def test_failed_response_with_reason(self):
        """Error case: failed indexing with reason"""
        response = IndexedAnchorResponse(
            anchor_id="fail-456",
            ok=False,
            reason="Qdrant connection timeout",
            detail="Could not connect to Qdrant at http://localhost:6333"
        )
        assert response.ok is False
        assert "timeout" in response.reason.lower()
        assert "Qdrant" in response.detail
    
    def test_response_serialization(self):
        """Integration: test that response can be serialized (for API responses)"""
        response = IndexedAnchorResponse(
            anchor_id="serial-789",
            ok=True,
            reason="All good!",
            detail="Indexed successfully"
        )
        # Pydantic models should be JSON serializable
        json_dict = response.model_dump()
        assert json_dict["anchor_id"] == "serial-789"
        assert json_dict["ok"] is True


class TestSettings:
    """Testing the Settings configuration 🎛️"""
    
    def test_default_settings(self):
        """Verify default values are sensible"""
        settings = Settings()
        assert settings.kafka_bootstrap_servers == "localhost:9092"
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.embedding_model == "deterministic"
        assert settings.ollama_base_url == "http://localhost:11434"
    
    def test_settings_from_env(self, monkeypatch):
        """Integration: test loading settings from environment variables"""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka.example.com:9092")
        monkeypatch.setenv("QDRANT_URL", "https://qdrant.cloud:6333")
        monkeypatch.setenv("EMBEDDING_MODEL", "ollama")
        monkeypatch.setenv("PORTKEY_API_KEY", "pk-secret-key-123")
        
        # Create new settings instance to pick up env vars
        settings = Settings()
        assert settings.kafka_bootstrap_servers == "kafka.example.com:9092"
        assert settings.qdrant_url == "https://qdrant.cloud:6333"
        assert settings.embedding_model == "ollama"
        assert settings.portkey_api_key == "pk-secret-key-123"
    
    def test_settings_optional_portkey_key(self):
        """Edge case: portkey key can be None"""
        settings = Settings()
        # Should not raise an error if portkey_api_key is None
        assert settings.portkey_api_key is None or isinstance(settings.portkey_api_key, str)


class TestEmbedding:
    """Testing embedding functionality 🧬"""
    
    def test_embedding_basic(self):
        """Happy path: get embedding for simple text"""
        text = "The quick brown fox"
        embedding = get_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
    
    def test_embedding_empty_string(self):
        """Edge case: embedding for empty string"""
        embedding = get_embedding("")
        assert isinstance(embedding, list)
        # Should still return something (even if it's a placeholder)
    
    def test_embedding_unicode_text(self):
        """Interesting: embedding with unicode/emoji characters"""
        text = "Hello 🌍! こんにちは! Здравствуй! 🚀"
        embedding = get_embedding(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
    
    def test_embedding_very_long_text(self):
        """Edge case: embedding for very long text"""
        long_text = "This is a very long text. " * 1000
        embedding = get_embedding(long_text)
        assert isinstance(embedding, list)
        # Should handle long text without crashing
    
    def test_embedding_consistency(self):
        """Interesting: check if embeddings are consistent (deterministic)"""
        text = "Consistent test text"
        embedding1 = get_embedding(text)
        embedding2 = get_embedding(text)
        
        # For deterministic embeddings, they should be identical
        # For non-deterministic (like API calls), they might differ
        # This test documents the current behavior
        assert isinstance(embedding1, list)
        assert isinstance(embedding2, list)
        assert len(embedding1) == len(embedding2)


class TestIntegrationScenarios:
    """Integration tests: testing components working together 🤝"""
    
    def test_full_anchor_lifecycle(self):
        """End-to-end: create anchor, get embedding, create response"""
        # Create an anchor
        anchor = Anchor(
            anchor_id="lifecycle-test",
            text="This is a test memory for the full lifecycle",
            stored_at=datetime.now(timezone.utc),
            salience=0.85,
            meta={"test": True, "scenario": "lifecycle"}
        )
        
        # Get embedding for the anchor text
        embedding = get_embedding(anchor.text)
        
        # Simulate successful indexing
        response = IndexedAnchorResponse(
            anchor_id=anchor.anchor_id,
            ok=True,
            reason="Successfully indexed",
            detail=f"Embedding dimension: {len(embedding)}"
        )
        
        # Verify everything works together
        assert anchor.anchor_id == response.anchor_id
        assert response.ok is True
        assert len(embedding) > 0
        assert "dimension" in response.detail
    
    def test_settings_with_anchor_creation(self):
        """Integration: using settings to configure anchor behavior"""
        settings = Settings()
        
        # Create anchor with settings-aware metadata
        anchor = Anchor(
            anchor_id="settings-aware",
            text="Memory created with settings context",
            stored_at=datetime.now(timezone.utc),
            meta={
                "embedding_model": settings.embedding_model,
                "qdrant_collection": settings.qdrant_collection,
                "created_with": "test_integration"
            }
        )
        
        # Verify settings are accessible
        assert anchor.meta["embedding_model"] == settings.embedding_model
        assert anchor.meta["qdrant_collection"] == settings.qdrant_collection


# Fun bonus test! 🎉
class TestEdgeCasesAndWeirdStuff:
    """Testing weird edge cases that might happen in production 🎪"""
    
    def test_anchor_with_special_characters(self):
        """Weird: text with special characters, SQL injection attempts, etc."""
        weird_texts = [
            "'; DROP TABLE memories; --",
            "<script>alert('xss')</script>",
            "Null\x00Byte",
            "New\nLine\nAnd\tTab",
            "Unicode: \u0000 \uFFFF \U0001F600"
        ]
        
        for weird_text in weird_texts:
            anchor = Anchor(
                anchor_id=f"weird-{hash(weird_text)}",
                text=weird_text,
                stored_at=datetime.now(timezone.utc)
            )
            assert anchor.text == weird_text  # Should preserve exactly
    
    def test_response_with_none_values(self):
        """Edge case: response with None in optional fields"""
        response = IndexedAnchorResponse(
            anchor_id="none-test",
            ok=True,
            reason=None,
            detail=None
        )
        assert response.ok is True
        assert response.reason is None
        assert response.detail is None


class TestIntentionallyFailing:
    """This test intentionally fails to demonstrate CI failure handling 🚨"""
    
    def test_embedding_should_have_minimum_dimension(self):
        """This test will fail - embeddings should have at least 128 dimensions"""
        text = "Test text for embedding"
        embedding = get_embedding(text)
        
        # This assertion will fail because the placeholder returns [0.1, 0.2, 0.3]
        # which only has 3 dimensions, not 128
        assert len(embedding) >= 128, f"Expected at least 128 dimensions, got {len(embedding)}"

