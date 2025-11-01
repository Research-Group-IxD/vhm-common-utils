# Tests for the embedding logic will go here.
from vhm_common_utils.embedding import get_embedding


def test_get_embedding_placeholder():
    """
    Tests the temporary placeholder implementation of get_embedding.
    This test can be removed or updated once the real logic is implemented.
    """
    # Given some dummy text
    dummy_text = "This is a test."

    # When get_embedding is called
    result = get_embedding(dummy_text)

    # Then the result should be the placeholder list
    assert result == [0.1, 0.2, 0.3]
