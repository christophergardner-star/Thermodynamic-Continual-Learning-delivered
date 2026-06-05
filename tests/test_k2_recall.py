"""Keystone #2 (K2.2b) — VectorVault prior-trial recall in the director.

The director surfaces similar past trials/results from vector memory onto its
experiment directives (advisory context, never a score change) so the loop builds
on prior work instead of blindly repeating it. Recall must be fully fail-safe:
construction failures, search failures, and empty queries all degrade to [] and
never raise into the live decision path.
"""
import tar_research_director as trd


class _FakeHit:
    def __init__(self, did, score, doc):
        self.document_id = did
        self.score = score
        self.document = doc


class _FakeVault:
    def __init__(self, ws):
        pass

    def search(self, query, n_results=3):
        hits = [
            _FakeHit("doc1", 0.42, "prior tcl split_cifar10 forgetting=0.13 directional"),
            _FakeHit("doc2", 0.31, "ewc baseline forgetting=0.20"),
            _FakeHit("doc3", 0.20, "si baseline forgetting=0.05"),
        ]
        return hits[:n_results]


def test_recall_returns_digests(tmp_path, monkeypatch):
    import tar_lab.memory as mem
    monkeypatch.setattr(mem, "VectorVault", _FakeVault)
    d = trd.ResearchDirector(tmp_path)
    hits = d._recall_prior_trials("catastrophic forgetting split cifar", n_results=2)
    assert len(hits) == 2
    assert hits[0]["document_id"] == "doc1"
    assert hits[0]["score"] == 0.42
    assert "tcl" in hits[0]["summary"]
    # vault is cached after first successful build
    assert d._vault is not None


def test_recall_failsafe_on_vault_error(tmp_path, monkeypatch):
    import tar_lab.memory as mem

    class _Boom:
        def __init__(self, ws):
            raise RuntimeError("chromadb unavailable")

    monkeypatch.setattr(mem, "VectorVault", _Boom)
    d = trd.ResearchDirector(tmp_path)
    assert d._recall_prior_trials("anything") == []
    # failure is cached so we don't retry-construct every cycle
    assert d._vault_failed is True
    assert d._recall_prior_trials("again") == []


def test_recall_failsafe_on_search_error(tmp_path, monkeypatch):
    import tar_lab.memory as mem

    class _SearchBoom:
        def __init__(self, ws):
            pass

        def search(self, query, n_results=3):
            raise RuntimeError("backend error")

    monkeypatch.setattr(mem, "VectorVault", _SearchBoom)
    d = trd.ResearchDirector(tmp_path)
    assert d._recall_prior_trials("x") == []


def test_recall_empty_query_returns_empty(tmp_path):
    d = trd.ResearchDirector(tmp_path)
    assert d._recall_prior_trials("") == []
    assert d._recall_prior_trials("   ") == []
