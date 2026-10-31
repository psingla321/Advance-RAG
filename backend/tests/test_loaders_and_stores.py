"""Document loaders, and the mapping logic of the server-backed vector stores.

The managed stores (Pinecone, Elasticsearch, ...) are exercised with fake clients:
what is verified here is the contract translation — payload shaping, filter
dialects and score handling — not the wire protocol of each service.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from app.rag.components.loaders import SUPPORTED_EXTENSIONS, strip_tags
from app.rag.components.vectorstores import (
    _flat_metadata, _milvus_filter, _odata_filter, _pinecone_filter, _unflatten,
)
from app.rag.registry import build, get_class, names
from app.rag.types import Chunk

# --------------------------------------------------------------------- loaders
AUTO = build("loader", "auto")


def test_notebook_keeps_markdown_and_fences_code() -> None:
    raw = json.dumps({
        "metadata": {"language_info": {"name": "python"}},
        "cells": [
            {"cell_type": "markdown", "source": ["# Travel\n", "Hotel cap is 350 USD."]},
            {"cell_type": "code", "source": ["print('hi')"]},
            {"cell_type": "code", "source": ["   "]},
        ],
    }).encode()
    sections = AUTO.load("policy.ipynb", raw)
    assert len(sections) == 2, "blank cells should be dropped"
    assert "Hotel cap is 350 USD." in sections[0].text
    assert sections[1].text == "```python\nprint('hi')\n```"


def test_notebook_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid notebook JSON"):
        AUTO.load("broken.ipynb", b"{not json")


def test_email_extracts_headers_and_body() -> None:
    raw = (b"From: a@x.com\r\nTo: b@y.com\r\nSubject: Hotel limit\r\n"
           b"Content-Type: text/plain\r\n\r\nThe cap is 350 USD in London.\r\n")
    sections = AUTO.load("mail.eml", raw)
    assert sections[0].heading == "Hotel limit"
    assert "From: a@x.com" in sections[0].text
    assert "350 USD in London" in sections[1].text


def test_email_reduces_html_body_to_text() -> None:
    raw = (b"Subject: Policy\r\nContent-Type: text/html\r\n\r\n"
           b"<html><body><p>Cap is <b>350</b> USD.</p><script>evil()</script></body></html>")
    text = AUTO.load("mail.eml", raw)[-1].text
    assert "350" in text and "evil()" not in text and "<p>" not in text


def test_xml_renders_rows_as_tag_value_lines() -> None:
    raw = b"<rows><row><city>London</city><cap>350</cap></row><row><city>Paris</city></row></rows>"
    text = AUTO.load("data.xml", raw)[0].text
    assert "city: London" in text and "cap: 350" in text


def test_xml_refuses_entity_declarations() -> None:
    # Billion-laughs / XXE vector: entity declarations are rejected, not expanded.
    with pytest.raises(ValueError, match="entities"):
        AUTO.load("evil.xml", b'<!DOCTYPE t [<!ENTITY a "boom">]><t>&a;</t>')


def test_epub_reads_chapters_and_drops_scripts() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ch1.xhtml", "<html><title>Chapter 1</title><body><p>Cap is 350 USD.</p>"
                                      "<script>bad()</script></body></html>")
        archive.writestr("cover.png", b"\x89PNG")
    sections = AUTO.load("book.epub", buffer.getvalue())
    assert len(sections) == 1 and sections[0].heading == "Chapter 1"
    assert "350 USD" in sections[0].text and "bad()" not in sections[0].text


def test_epub_rejects_unreadable_archive() -> None:
    with pytest.raises(ValueError, match="not a readable EPUB"):
        AUTO.load("book.epub", b"this is not a zip")


def test_rtf_strips_control_words_and_tables() -> None:
    raw = (rb"{\rtf1\ansi{\fonttbl\f0 Arial;}{\colortbl;\red0\green0\blue0;}"
           rb"\f0\fs24 Hotel cap is 350 USD.\par Second line.\par}")
    text = AUTO.load("doc.rtf", raw)[0].text
    assert text == "Hotel cap is 350 USD.\nSecond line."
    assert "Arial" not in text and "fonttbl" not in text


def test_rtf_decodes_escapes() -> None:
    # \'e9 is a cp1252 byte escape; 荤 is the RTF unicode escape for the euro sign.
    raw = rb"{\rtf1 caf\'e9 and \u" + b"8364 ?}"
    text = AUTO.load("doc.rtf", raw)[0].text
    assert "café" in text and "€" in text


def test_source_files_route_to_the_text_loader() -> None:
    for filename in ("app.ts", "Main.java", "lib.go", "main.rs", "deploy.tf"):
        sections = AUTO.load(filename, b"const x = 1")
        assert sections[0].text == "const x = 1", filename


def test_every_loader_extension_is_walkable() -> None:
    """A format the auto loader handles must also be picked up by directory ingestion."""
    for name in names("loader"):
        if name == "auto":
            continue
        for extension in getattr(get_class("loader", name), "extensions", ()):
            assert extension in SUPPORTED_EXTENSIONS, f"{name} handles {extension}, but iter_files skips it"


def test_strip_tags_inserts_breaks() -> None:
    assert strip_tags("<p>One</p><p>Two</p>") == "One\n\nTwo"


# ---------------------------------------------------------------- store mapping
def _chunk() -> Chunk:
    return Chunk(id="c1", text="Cap is 350 USD.", doc_id="d1", filename="travel.md",
                 heading="Hotels", ord=2, metadata={"acl": ["hr"], "year": 2025})


def test_flat_metadata_round_trips_nested_metadata() -> None:
    flat = _flat_metadata(_chunk())
    assert isinstance(flat["metadata"], str), "nested dicts must be serialised for scalar-only stores"
    restored = _unflatten(flat)
    assert restored.id == "c1" and restored.metadata["acl"] == ["hr"] and restored.ord == 2


def test_unflatten_survives_corrupt_metadata() -> None:
    assert _unflatten({"id": "x", "metadata": "{not json"}).metadata == {}


def test_filter_dialects() -> None:
    generic = {"doc_id": "d1", "acl": ["hr", "all"]}
    assert _pinecone_filter(generic) == {"doc_id": {"$eq": "d1"}, "acl": {"$in": ["hr", "all"]}}
    assert _milvus_filter(generic) == 'doc_id in ["d1"] and acl in ["hr", "all"]'
    odata = _odata_filter(generic)
    assert "doc_id eq 'd1'" in odata and "acl/any(g: g eq 'hr')" in odata
    assert _pinecone_filter(None) is None and _milvus_filter(None) == "" and _odata_filter(None) is None


def test_odata_filter_escapes_quotes() -> None:
    assert "O''Brien" in _odata_filter({"filename": "O'Brien"})


class FakePineconeIndex:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.deleted: list[dict] = []

    def upsert(self, vectors, namespace):  # noqa: ANN001 - fake mirrors the SDK
        for row in vectors:
            self.rows[row["id"]] = row

    def query(self, vector, top_k, namespace, include_metadata, filter):  # noqa: A002, ANN001
        matches = [{"id": r["id"], "score": 0.9, "metadata": r["metadata"]} for r in self.rows.values()]
        return {"matches": matches[:top_k]}

    def delete(self, filter, namespace):  # noqa: A002, ANN001
        self.deleted.append(filter)


async def test_pinecone_store_maps_payload_and_scores() -> None:
    index = FakePineconeIndex()
    store = build("vectorstore", {"type": "pinecone", "client": index})
    await store.init(3)
    await store.upsert([_chunk()], [[0.1, 0.2, 0.3]], "ns")

    assert index.rows["c1"]["values"] == [0.1, 0.2, 0.3]
    results = await store.search([0.1, 0.2, 0.3], 5, "ns")
    assert len(results) == 1
    assert results[0].chunk.text == "Cap is 350 USD." and results[0].chunk.metadata["acl"] == ["hr"]
    assert results[0].score == pytest.approx(0.9) and results[0].rank == 1

    await store.delete(["d1"], "ns")
    assert index.deleted == [{"doc_id": {"$in": ["d1"]}}]


class FakeElasticsearch:
    def __init__(self) -> None:
        self.operations: list[dict] = []
        self.searched: dict = {}
        self.indices = self

    async def exists(self, index):  # noqa: ANN001
        return False

    async def create(self, index, body):  # noqa: ANN001
        self.created = body

    async def bulk(self, operations, refresh):  # noqa: ANN001
        self.operations = operations

    async def search(self, index, size=None, knn=None, body=None):  # noqa: ANN001
        self.searched = {"knn": knn, "body": body}
        return {"hits": {"hits": [{"_id": "c1", "_score": 1.4,
                                   "_source": _flat_metadata(_chunk())}]}}


async def test_elasticsearch_store_builds_knn_query_with_filters() -> None:
    client = FakeElasticsearch()
    store = build("vectorstore", {"type": "elasticsearch", "client": client})
    await store.init(3)
    await store.upsert([_chunk()], [[0.1, 0.2, 0.3]], "ns")

    # bulk() takes alternating action/document lines.
    assert client.operations[0]["index"]["_id"] == "c1"
    assert client.operations[1]["vector"] == [0.1, 0.2, 0.3]

    results = await store.search([0.1, 0.2, 0.3], 4, "ns", {"acl": ["hr"]})
    assert client.searched["knn"]["field"] == "vector"
    assert client.searched["knn"]["filter"] == [{"terms": {"acl": ["hr"]}}]
    assert results[0].score == pytest.approx(1.4) and results[0].chunk.heading == "Hotels"


async def test_opensearch_uses_its_own_query_shape() -> None:
    client = FakeElasticsearch()
    store = build("vectorstore", {"type": "elasticsearch", "client": client, "opensearch": True})
    await store.init(3)
    await store.search([0.1, 0.2, 0.3], 4, "ns")
    assert client.searched["body"]["query"] == {"knn": {"vector": {"vector": [0.1, 0.2, 0.3], "k": 4}}}


def test_every_store_declares_its_dependency() -> None:
    """Only the dependency-free stores may advertise no pip extra."""
    from app.rag.registry import list_components

    free = {c["name"] for c in list_components("vectorstore")["vectorstore"] if not c["requires"]}
    assert free == {"memory"}
