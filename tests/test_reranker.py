from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Block heavy transitive imports from src.__init__ -> src.pipeline -> kuzu, etc.
# We only need src.retrieval.rrf and src.retrieval.reranker.
# ---------------------------------------------------------------------------
_fake = types.ModuleType("kuzu")
_fake.Database = MagicMock
_fake.Schema = MagicMock
sys.modules.setdefault("kuzu", _fake)

_fake_lancedb = types.ModuleType("lancedb")
_fake_lancedb.connect = MagicMock
sys.modules.setdefault("lancedb", _fake_lancedb)

_fake_st = types.ModuleType("sentence_transformers")
_fake_st.CrossEncoder = MagicMock
sys.modules.setdefault("sentence_transformers", _fake_st)

# Prevent src.__init__ from pulling the whole app
sys.modules.setdefault("src.pipeline", types.ModuleType("src.pipeline"))
sys.modules["src.pipeline"].FinancialGraphRAGPipeline = MagicMock  # type: ignore[attr-defined]

# Mock heavy modules imported by src.retrieval.__init__
_fake_dense = types.ModuleType("src.retrieval.dense")
_fake_dense.DenseRetriever = MagicMock
sys.modules.setdefault("src.retrieval.dense", _fake_dense)

_fake_sparse = types.ModuleType("src.retrieval.sparse")
_fake_sparse.SparseRetriever = MagicMock
sys.modules.setdefault("src.retrieval.sparse", _fake_sparse)

_fake_gt = types.ModuleType("src.retrieval.graph_traversal")
_fake_gt.GraphTraversalRetriever = MagicMock
sys.modules.setdefault("src.retrieval.graph_traversal", _fake_gt)

_fake_gen = types.ModuleType("src.retrieval.generator")
_fake_gen.ResponseGenerator = MagicMock
sys.modules.setdefault("src.retrieval.generator", _fake_gen)

_fake_rp = types.ModuleType("src.retrieval.pipeline")
_fake_rp.RetrievalPipeline = MagicMock
_fake_rp.RetrievalResult = MagicMock
sys.modules.setdefault("src.retrieval.pipeline", _fake_rp)

import pytest  # noqa: E402

from src.retrieval.reranker import (  # noqa: E402
    CrossEncoderReranker,
    RerankedResult,
    _detect_device,
    _jaccard,
    _sigmoid,
    _tokenize,
)
from src.retrieval.rrf import FusedResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _fused(text: str, chunk_id: str = "", **meta) -> FusedResult:
    return FusedResult(
        chunk_id=chunk_id or text[:8],
        text=text,
        rrf_score=1.0,
        metadata=meta,
    )


def _make_reranker(scores: list[float]):
    """Return a CrossEncoderReranker with a mocked model.predict."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "mock-model"
    reranker._device = "cpu"

    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    reranker._model = mock_model
    return reranker


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)

    def test_known_value(self):
        assert _sigmoid(1.0) == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == {"hello", "world"}

    def test_numbers(self):
        assert _tokenize("rev 2023 $1.2B") == {"rev", "2023", "1", "2b"}

    def test_empty(self):
        assert _tokenize("") == set()

    def test_punctuation_only(self):
        assert _tokenize("!!!???") == set()


class TestJaccard:
    def test_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        assert _jaccard(a, b) == pytest.approx(2.0 / 4.0)

    def test_empty_left(self):
        assert _jaccard(set(), {"a"}) == pytest.approx(0.0)

    def test_empty_right(self):
        assert _jaccard({"a"}, set()) == pytest.approx(0.0)

    def test_both_empty(self):
        assert _jaccard(set(), set()) == pytest.approx(0.0)


class TestDetectDevice:
    def test_preferred_cuda(self):
        assert _detect_device("cuda") == "cuda"

    def test_cpu_explicit(self):
        assert _detect_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# CrossEncoderReranker.rerank (mocked model)
# ---------------------------------------------------------------------------

class TestRerankerRerank:
    def test_empty_candidates(self):
        r = _make_reranker([])
        assert r.rerank("query", [], top_k=5) == []

    def test_returns_top_k(self):
        candidates = [_fused(f"chunk-{i}", chunk_id=f"c{i}") for i in range(20)]
        scores = [float(i) / 20.0 for i in range(20)]
        r = _make_reranker(scores)
        result = r.rerank("q", candidates, top_k=5)
        assert len(result) == 5

    def test_ordering_by_score(self):
        candidates = [_fused("low"), _fused("mid"), _fused("high")]
        scores = [0.1, 0.5, 0.9]
        r = _make_reranker(scores)
        result = r.rerank("q", candidates, top_k=3)
        texts = [res.text for res in result]
        assert texts == ["high", "mid", "low"]

    def test_min_score_filter(self):
        candidates = [_fused("a"), _fused("b")]
        scores = [0.9, -2.0]  # sigmoid(-2) ≈ 0.12 < 0.2
        r = _make_reranker(scores)
        result = r.rerank("q", candidates, top_k=10, min_score=0.2)
        texts = [res.text for res in result]
        assert "b" not in texts

    def test_returns_reranked_result_type(self):
        candidates = [_fused("chunk", chunk_id="id1", company_ticker="AAPL")]
        scores = [0.8]
        r = _make_reranker(scores)
        result = r.rerank("q", candidates, top_k=1)
        assert len(result) == 1
        rr = result[0]
        assert isinstance(rr, RerankedResult)
        assert rr.chunk_id == "id1"
        assert rr.score == pytest.approx(_sigmoid(0.8))
        assert rr.metadata["company_ticker"] == "AAPL"

    def test_diversity_penalty(self):
        """Two near-identical chunks should get a diversity penalty; two
        completely different chunks should not."""
        a = _fused("the cat sat on the mat", chunk_id="a")
        b = _fused("completely unrelated content here", chunk_id="b")
        c = _fused("the cat sat on the mat also", chunk_id="c")

        # Same scores for all – diversity should break the tie
        r = _make_reranker([0.8, 0.8, 0.8])
        result = r.rerank(
            "q",
            [a, b, c],
            top_k=2,
            diversity_lambda=0.5,
            min_score=0.0,
        )
        ids = [res.chunk_id for res in result]
        # 'b' should be picked before the second copy of cat-mat
        assert ids[0] in ("a", "c")
        assert ids[1] == "b"

    def test_diversity_lambda_1_no_penalty(self):
        """With lambda=1.0 diversity has no effect; pure score ordering."""
        a = _fused("similar text one", chunk_id="a")
        b = _fused("similar text two", chunk_id="b")
        r = _make_reranker([0.9, 0.8])
        result = r.rerank(
            "q",
            [a, b],
            top_k=2,
            diversity_lambda=1.0,
            min_score=0.0,
        )
        assert [res.chunk_id for res in result] == ["a", "b"]

    def test_candidates_fewer_than_top_k(self):
        candidates = [_fused("only"), _fused("two")]
        r = _make_reranker([0.5, 0.3])
        result = r.rerank("q", candidates, top_k=10)
        assert len(result) == 2
