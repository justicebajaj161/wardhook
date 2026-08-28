"""Tests for chunking, loading, embedding, storage, and retrieval."""

from __future__ import annotations

import pytest

from wardhook.core.rag.chunking import Chunk, chunk_text
from wardhook.core.rag.embeddings import HashingEmbeddings, cosine_similarity, resolve_embeddings
from wardhook.core.rag.loaders import (
    Document,
    DocumentLoadError,
    load_directory,
    load_document,
    load_to_chunks,
)
from wardhook.core.rag.retriever import Retriever, format_citations
from wardhook.core.rag.store import InMemoryVectorStore


def _minimal_pdf(*pages: str) -> bytes:
    """Build a valid single-font PDF with one text-bearing page per argument.

    Written by hand rather than fixtured as a binary file so the PDF path is
    exercised against real ``pypdf`` parsing without carrying a checked-in
    binary that nobody can review in a diff.
    """
    streams = [f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode() for text in pages]
    page_ids = [3 + i for i in range(len(streams))]
    content_ids = [3 + len(streams) + 1 + i for i in range(len(streams))]
    font_id = 3 + len(streams)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{i} 0 R" for i in page_ids), len(streams)
        ).encode(),
    ]
    for content_id in content_ids:
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for stream in streams:
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


class TestChunking:
    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text("A short note.", "note.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "A short note."
        assert chunks[0].source == "note.txt"
        assert chunks[0].chunk_index == 0

    def test_empty_and_whitespace_input_yields_nothing(self):
        assert chunk_text("", "x.txt") == []
        assert chunk_text("   \n\n  ", "x.txt") == []

    def test_indices_are_sequential(self):
        text = "\n\n".join(f"Paragraph number {i} with some filler." for i in range(10))
        chunks = chunk_text(text, "doc.md", chunk_size=60, chunk_overlap=0)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_prefers_paragraph_boundaries_over_mid_sentence_cuts(self):
        chunks = chunk_text("First para.\n\nSecond para.", "d.md", chunk_size=20, chunk_overlap=0)
        assert [c.text.strip() for c in chunks] == ["First para.", "Second para."]

    def test_overlap_repeats_trailing_context(self):
        text = "abcdefghij" * 10
        chunks = chunk_text(text, "d.txt", chunk_size=40, chunk_overlap=10)
        assert len(chunks) > 1
        assert chunks[1].text[:10] == chunks[0].text[-10:]

    def test_text_with_no_separators_still_terminates(self):
        chunks = chunk_text("x" * 250, "blob.txt", chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 3
        assert all(len(c.text) <= 100 for c in chunks)

    def test_metadata_is_copied_to_every_chunk_independently(self):
        chunks = chunk_text("a\n\nb\n\nc", "d.md", chunk_size=5, chunk_overlap=0, metadata={"k": 1})
        assert all(c.metadata == {"k": 1} for c in chunks)
        chunks[0].metadata["k"] = 999
        assert chunks[1].metadata["k"] == 1, "chunks must not share one metadata dict"

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 200)],
    )
    def test_invalid_parameters_are_rejected(self, size, overlap):
        with pytest.raises(ValueError):
            chunk_text("some text", "d.txt", chunk_size=size, chunk_overlap=overlap)

    def test_round_trips_through_dict(self):
        original = Chunk("body", "s.md", 2, {"page": 7})
        assert Chunk.from_dict(original.to_dict()) == original

    def test_id_combines_source_and_index(self):
        assert Chunk("t", "policy.md", 3).id == "policy.md#3"


class TestLoaders:
    def test_loads_text_and_markdown(self, tmp_path):
        (tmp_path / "a.txt").write_text("plain text", encoding="utf-8")
        (tmp_path / "b.md").write_text("# heading", encoding="utf-8")
        assert load_document(tmp_path / "a.txt").text == "plain text"
        assert load_document(tmp_path / "b.md").metadata["file_type"] == "md"

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nope.txt")

    def test_directory_passed_to_load_document_is_rejected(self, tmp_path):
        with pytest.raises(DocumentLoadError, match="directory"):
            load_document(tmp_path)

    def test_unsupported_suffix_is_rejected_with_a_helpful_message(self, tmp_path):
        path = tmp_path / "data.parquet"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(DocumentLoadError, match="Unsupported file type"):
            load_document(path)

    def test_non_utf8_file_raises_documentloaderror(self, tmp_path):
        path = tmp_path / "latin.txt"
        path.write_bytes(b"caf\xe9 latin-1 bytes")
        with pytest.raises(DocumentLoadError, match="UTF-8"):
            load_document(path)

    def test_directory_load_is_recursive_and_sorted(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "b.txt").write_text("second", encoding="utf-8")
        (tmp_path / "a.md").write_text("first", encoding="utf-8")
        (tmp_path / "sub" / "c.txt").write_text("third", encoding="utf-8")
        (tmp_path / "ignored.parquet").write_text("skip me", encoding="utf-8")

        docs = load_directory(tmp_path)
        assert [d.source.split("/")[-1] for d in docs] == ["a.md", "b.txt", "c.txt"]

    def test_directory_load_can_be_non_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("top", encoding="utf-8")
        (tmp_path / "sub" / "b.txt").write_text("nested", encoding="utf-8")
        assert len(load_directory(tmp_path, recursive=False)) == 1

    def test_directory_load_skips_unreadable_files_by_default(self, tmp_path):
        (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe invalid")
        assert len(load_directory(tmp_path)) == 1

    def test_directory_load_can_fail_loudly_instead(self, tmp_path):
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe invalid")
        with pytest.raises(DocumentLoadError):
            load_directory(tmp_path, skip_errors=False)

    def test_not_a_directory_raises(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            load_directory(path)

    def test_loads_a_pdf_and_records_the_page_count(self, tmp_path):
        path = tmp_path / "policy.pdf"
        path.write_bytes(_minimal_pdf("Storm damage carries a 500 excess."))

        doc = load_document(path)
        assert "Storm damage carries a 500 excess." in doc.text
        assert doc.metadata == {"file_type": "pdf", "page_count": 1}

    def test_pdf_pages_are_marked_so_a_chunk_keeps_its_page(self, tmp_path):
        # The page marker is the only thing tying a mid-document chunk back to
        # a page number once chunking has thrown the boundaries away.
        path = tmp_path / "manual.pdf"
        path.write_bytes(_minimal_pdf("First page body.", "Second page body."))

        doc = load_document(path)
        assert "[page 1]" in doc.text
        assert "[page 2]" in doc.text
        assert doc.text.index("[page 1]") < doc.text.index("[page 2]")
        assert doc.metadata["page_count"] == 2

    def test_a_pdf_with_no_extractable_text_loads_as_empty(self, tmp_path):
        # A scanned PDF is pages of images. Loading must not crash; it yields
        # nothing to chunk, which is the honest answer without OCR.
        path = tmp_path / "scan.pdf"
        path.write_bytes(_minimal_pdf(""))

        doc = load_document(path)
        assert doc.text == ""
        assert doc.metadata["page_count"] == 1

    def test_a_corrupt_pdf_raises_documentloaderror(self, tmp_path):
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not actually a pdf")
        with pytest.raises(DocumentLoadError, match="Could not read PDF"):
            load_document(path)

    def test_a_pdf_is_picked_up_by_a_directory_load(self, tmp_path):
        (tmp_path / "notes.txt").write_text("plain", encoding="utf-8")
        (tmp_path / "policy.pdf").write_bytes(_minimal_pdf("Excess applies."))

        docs = load_directory(tmp_path)
        assert sorted(d.metadata["file_type"] for d in docs) == ["pdf", "txt"]

    def test_document_metadata_survives_chunking(self):
        docs = [Document(text="a b c", source="memo.txt", metadata={"team": "risk"})]
        chunks = load_to_chunks(docs, chunk_size=100, chunk_overlap=0)
        assert chunks[0].metadata["team"] == "risk"
        assert chunks[0].source == "memo.txt"


class TestEmbeddings:
    def test_is_deterministic_across_instances(self):
        # Guards the BLAKE2b choice: Python's built-in hash() is salted per
        # process, which would make a saved index unsearchable after restart.
        a = HashingEmbeddings(dim=64).embed_query("insurance premium")
        b = HashingEmbeddings(dim=64).embed_query("insurance premium")
        assert a == b

    def test_is_order_insensitive_and_topic_sensitive(self):
        e = HashingEmbeddings(dim=128)
        same = cosine_similarity(
            e.embed_query("annual insurance premium"),
            e.embed_query("premium insurance annual"),
        )
        different = cosine_similarity(
            e.embed_query("annual insurance premium"),
            e.embed_query("chocolate cake recipe"),
        )
        assert same == pytest.approx(1.0)
        assert different < 0.2

    def test_vectors_are_unit_length(self):
        vector = HashingEmbeddings(dim=32).embed_query("some words here")
        assert sum(v * v for v in vector) == pytest.approx(1.0)

    def test_text_without_tokens_gives_the_zero_vector(self):
        assert HashingEmbeddings(dim=16).embed_query("!!! ???") == [0.0] * 16

    def test_batch_matches_single(self):
        e = HashingEmbeddings(dim=32)
        assert e.embed_documents(["one", "two"]) == [e.embed_query("one"), e.embed_query("two")]

    def test_rejects_non_positive_dimension(self):
        with pytest.raises(ValueError):
            HashingEmbeddings(dim=0)

    def test_cosine_similarity_edges(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_resolve_rejects_an_incomplete_object(self):
        class HalfDone:
            def embed_query(self, text): ...

        with pytest.raises(TypeError, match="embed_documents"):
            resolve_embeddings(HalfDone())

    def test_resolve_defaults_to_the_offline_backend(self):
        assert isinstance(resolve_embeddings(None), HashingEmbeddings)


class TestVectorStore:
    def test_empty_store_searches_cleanly(self):
        assert InMemoryVectorStore().search("anything") == []
        assert len(InMemoryVectorStore()) == 0

    def test_add_returns_chunk_ids(self, sample_store):
        ids = sample_store.add(chunk_text("New content here.", "new.md"))
        assert ids == ["new.md#0"]

    def test_retrieves_the_topically_correct_document(self, sample_store):
        hits = sample_store.search("what excess applies to storm damage?", k=1)
        assert hits[0].chunk.source == "policy.md"

    def test_results_are_ordered_by_descending_score(self, sample_store):
        hits = sample_store.search("storm damage excess", k=2)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_k_larger_than_corpus_returns_everything(self, sample_store):
        assert len(sample_store.search("anything", k=99)) == len(sample_store)

    def test_non_positive_k_returns_nothing(self, sample_store):
        assert sample_store.search("storm", k=0) == []

    def test_query_without_tokens_returns_nothing(self, sample_store):
        assert sample_store.search("!!!") == []

    def test_chunk_without_tokens_never_outranks_a_real_match(self):
        store = InMemoryVectorStore()
        store.add([*chunk_text("!!! ???", "junk.md"), *chunk_text("storm excess", "policy.md")])
        assert store.search("storm excess", k=1)[0].chunk.source == "policy.md"

    def test_mismatched_embedding_width_is_rejected(self):
        store = InMemoryVectorStore(embeddings=HashingEmbeddings(dim=32))
        store.add(chunk_text("first", "a.md"))
        store.embeddings = HashingEmbeddings(dim=64)
        with pytest.raises(ValueError, match="width mismatch"):
            store.add(chunk_text("second", "b.md"))

    def test_search_result_converts_to_a_citation(self, sample_store):
        citation = sample_store.search("storm damage", k=1)[0].to_citation()
        assert set(citation) == {"source", "chunk_index", "score", "text", "metadata"}
        assert citation["source"] == "policy.md"

    def test_save_and_load_round_trip(self, tmp_path, sample_store):
        sidecar = sample_store.save(tmp_path / "index")
        assert sidecar.exists()
        assert (tmp_path / "index.npz").exists()

        restored = InMemoryVectorStore.load(tmp_path / "index")
        assert len(restored) == len(sample_store)
        assert restored.search("storm damage excess", k=1)[0].chunk.source == "policy.md"

    def test_saving_an_empty_store_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            InMemoryVectorStore().save(tmp_path / "index")

    def test_loading_a_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InMemoryVectorStore.load(tmp_path / "absent")

    def test_desynchronised_index_files_are_detected(self, tmp_path, sample_store):
        import json

        sidecar = sample_store.save(tmp_path / "index")
        payload = json.loads(sidecar.read_text())
        payload["chunks"].pop()
        sidecar.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="out of sync"):
            InMemoryVectorStore.load(tmp_path / "index")


class TestRetriever:
    def test_returns_citation_records(self, sample_store):
        citations = Retriever(sample_store, k=1).retrieve("storm damage excess")
        assert citations[0]["source"] == "policy.md"
        assert citations[0]["chunk_index"] == 0

    def test_threshold_filters_weak_matches(self, sample_store):
        assert Retriever(sample_store, k=2, score_threshold=0.99).retrieve("storm") == []

    def test_context_block_and_citations_stay_aligned(self, sample_store):
        block, citations = Retriever(sample_store, k=2).context_for("storm damage")
        for position, citation in enumerate(citations, start=1):
            assert f"[{position}] {citation['source']}" in block

    def test_rejects_an_object_that_is_not_a_store(self):
        with pytest.raises(TypeError, match=r"no.*search"):
            Retriever(object())

    def test_rejects_non_positive_k(self, sample_store):
        with pytest.raises(ValueError):
            Retriever(sample_store, k=0)

    def test_formats_an_empty_citation_list_as_empty_string(self):
        assert format_citations([]) == ""

    def test_format_tolerates_partial_records(self):
        assert "[1] unknown (chunk 0)" in format_citations([{"text": "body"}])


class TestReprs:
    """Debug representations are what a developer sees first in a REPL or log."""

    def test_store_repr_reports_its_size(self):
        store = InMemoryVectorStore()
        store.add(chunk_text("some text", "a.txt"))
        assert "InMemoryVectorStore(chunks=1" in repr(store)

    def test_embeddings_repr_reports_the_dimension(self):
        assert repr(HashingEmbeddings(dim=64)) == "HashingEmbeddings(dim=64)"

    def test_retriever_repr_reports_its_thresholds(self):
        retriever = Retriever(InMemoryVectorStore(), k=3, score_threshold=0.2)
        assert repr(retriever) == "Retriever(k=3, score_threshold=0.2)"


class TestStoreEdges:
    def test_chunks_property_returns_a_copy_in_insertion_order(self):
        store = InMemoryVectorStore()
        store.add(chunk_text("first document", "a.txt"))
        store.add(chunk_text("second document", "b.txt"))

        exposed = store.chunks
        exposed.clear()
        assert [c.source for c in store.chunks] == ["a.txt", "b.txt"]

    def test_adding_nothing_is_a_no_op(self):
        store = InMemoryVectorStore()
        assert store.add([]) == []
        assert len(store) == 0

    def test_a_one_dimensional_embedding_is_rejected(self):
        # A flat vector silently broadcasts against the matrix and produces
        # meaningless similarity scores, so it is refused at the door.
        class FlatEmbeddings:
            dim = 4

            def embed_documents(self, texts):
                return [0.1, 0.2, 0.3, 0.4]

            def embed_query(self, text):
                return [0.1, 0.2, 0.3, 0.4]

        store = InMemoryVectorStore(embeddings=FlatEmbeddings())
        with pytest.raises(ValueError, match="2-D sequence of vectors"):
            store.add(chunk_text("text", "a.txt"))

    def test_trailing_whitespace_does_not_become_an_empty_chunk(self):
        # A document ending in blank space leaves a whitespace-only buffer. It
        # must be dropped: an empty chunk embeds to noise and pollutes results.
        chunks = chunk_text("aaa bbb   ", "a.txt", chunk_size=4, chunk_overlap=0)
        assert [c.text for c in chunks] == ["aaa ", "bbb "]
        assert all(c.text.strip() for c in chunks)

    def test_a_chunk_shorter_than_the_overlap_still_terminates(self):
        # The overlap tail is re-seeded on every boundary; a chunk_size smaller
        # than the overlap must not loop or drop the remainder.
        chunks = chunk_text("alpha beta gamma delta", "a.txt", chunk_size=8, chunk_overlap=6)
        assert chunks
        assert "".join(c.text for c in chunks).replace(" ", "").endswith("delta")
