"""bref_scraper 단위 테스트 (HTTP는 mock으로)."""
from io import StringIO
from unittest.mock import patch, MagicMock

import pandas as pd

from src.collector import bref_scraper


SAMPLE_HTML = """
<html><body>
<table id="teams_standard_pitching">
  <thead><tr><th>Team</th><th>IP</th><th>ER</th><th>ERA</th></tr></thead>
  <tbody>
    <tr><td>LAD</td><td>100</td><td>30</td><td>2.70</td></tr>
    <tr><td>NYY</td><td>120</td><td>40</td><td>3.00</td></tr>
  </tbody>
</table>
</body></html>
"""


class TestSafeBrefScrape:
    def test_success(self):
        mock_resp = MagicMock(status_code=200, text=SAMPLE_HTML)
        mock_resp.raise_for_status = MagicMock()
        with patch.object(bref_scraper.requests, "get", return_value=mock_resp), \
             patch.object(bref_scraper.time, "sleep"):  # skip waits
            df = bref_scraper.safe_bref_scrape("http://test.local/x")
        assert not df.empty
        assert "Team" in df.columns
        assert len(df) == 2

    def test_429_retry_then_success(self):
        mock_429 = MagicMock(status_code=429)
        mock_ok  = MagicMock(status_code=200, text=SAMPLE_HTML)
        mock_ok.raise_for_status = MagicMock()
        with patch.object(bref_scraper.requests, "get", side_effect=[mock_429, mock_ok]), \
             patch.object(bref_scraper.time, "sleep"):
            df = bref_scraper.safe_bref_scrape("http://test.local/x", max_retries=3)
        assert not df.empty

    def test_all_failures_returns_empty(self):
        with patch.object(bref_scraper.requests, "get", side_effect=Exception("network")), \
             patch.object(bref_scraper.time, "sleep"):
            df = bref_scraper.safe_bref_scrape("http://test.local/x", max_retries=2)
        assert df.empty


class TestUpdateBrefSeason:
    def test_writes_both_files(self, tmp_path):
        df = pd.read_html(StringIO(SAMPLE_HTML))[0]
        with patch.object(bref_scraper, "safe_bref_scrape", return_value=df):
            counts = bref_scraper.update_bref_season(2026, raw_dir=tmp_path)
        assert (tmp_path / "bref_pitching_2026.csv").exists()
        assert (tmp_path / "bref_batting_2026.csv").exists()
        assert counts["pitching"] == 2
        assert counts["batting"] == 2

    def test_empty_scrape_skips_write(self, tmp_path):
        empty_df = pd.DataFrame()
        with patch.object(bref_scraper, "safe_bref_scrape", return_value=empty_df):
            counts = bref_scraper.update_bref_season(2026, raw_dir=tmp_path)
        assert not (tmp_path / "bref_pitching_2026.csv").exists()
        assert counts["pitching"] == 0
