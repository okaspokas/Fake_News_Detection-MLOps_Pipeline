"""Tests for the data pipeline module."""

import os
import pandas as pd
import pytest

from src.data_pipeline import load_config, load_raw_data, clean_data


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def raw_data(config):
    return load_raw_data(config["data"]["raw_path"])


class TestDataPipeline:
    def test_config_loads(self, config):
        assert "data" in config
        assert "model" in config
        assert "raw_path" in config["data"]

    def test_raw_data_loads(self, raw_data):
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) > 0
        assert "title" in raw_data.columns
        assert "real" in raw_data.columns

    def test_clean_data_removes_nulls(self, raw_data):
        cleaned = clean_data(raw_data)
        assert cleaned["title"].isnull().sum() == 0
        assert cleaned["real"].isnull().sum() == 0

    def test_clean_data_removes_duplicates(self, raw_data):
        cleaned = clean_data(raw_data)
        assert len(cleaned) == len(cleaned.drop_duplicates())

    def test_cleaned_data_not_empty(self, raw_data):
        cleaned = clean_data(raw_data)
        assert len(cleaned) > 100
