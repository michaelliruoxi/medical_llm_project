"""Tests for src/ingest.py â€” XML parsing, text cleaning, deduplication."""

import pandas as pd

from src.ingest import parse_xml_file, _clean_text, deduplicate


SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<Document>
  <QAPair pid="1">
    <Question>What is diabetes?</Question>
    <Answer>Diabetes is a chronic condition affecting blood sugar.</Answer>
  </QAPair>
  <QAPair pid="2">
    <Question>What causes hypertension?</Question>
    <Answer>Hypertension can be caused by genetics, diet, and stress.</Answer>
  </QAPair>
</Document>
"""

MALFORMED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<Document>
  <QAPair>
    <Question>Incomplete XML
"""


class TestParseXML:
    def test_parse_valid_xml(self, tmp_path):
        xml_file = tmp_path / "test.xml"
        xml_file.write_text(SAMPLE_XML, encoding="utf-8")
        records = parse_xml_file(xml_file)
        assert len(records) == 2
        assert records[0]["question"] == "What is diabetes?"
        assert "blood sugar" in records[0]["answer"]
        assert records[0]["source"] == tmp_path.name

    def test_parse_malformed_xml(self, tmp_path):
        xml_file = tmp_path / "bad.xml"
        xml_file.write_text(MALFORMED_XML, encoding="utf-8")
        records = parse_xml_file(xml_file)
        assert records == []

    def test_parse_empty_qa(self, tmp_path):
        xml = '<?xml version="1.0"?><Document><QAPair><Question></Question><Answer></Answer></QAPair></Document>'
        xml_file = tmp_path / "empty.xml"
        xml_file.write_text(xml, encoding="utf-8")
        records = parse_xml_file(xml_file)
        assert records == []


class TestCleanText:
    def test_strip_html_tags(self):
        assert _clean_text("Hello <b>world</b>") == "Hello world"

    def test_strip_html_entities(self):
        assert _clean_text("A&amp;B") == "A B"

    def test_collapse_whitespace(self):
        assert _clean_text("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert _clean_text("") == ""


class TestDeduplicate:
    def test_removes_case_insensitive_duplicates(self):
        df = pd.DataFrame({
            "question": ["What is X?", "what is x?", "What is Y?"],
            "answer": ["A1", "A2", "A3"],
        })
        result = deduplicate(df)
        assert len(result) == 2
        assert result.iloc[0]["answer"] == "A1"  # keeps first

    def test_no_duplicates(self):
        df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q3"],
            "answer": ["A1", "A2", "A3"],
        })
        result = deduplicate(df)
        assert len(result) == 3
