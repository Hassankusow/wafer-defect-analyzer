"""
Tests for the Wafer Defect Clustering & Yield Impact Analyzer.
Run with: pytest test_wafer_analyzer.py -v
"""

import pytest
import numpy as np
import pandas as pd
from wafer_analyzer import (
    WaferConfig,
    simulate_defects,
    classify_defects,
    poisson_yield,
    murphy_yield,
    seeds_yield,
    compute_yield_analysis,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return WaferConfig(radius_mm=150, edge_exclusion_mm=3, die_size_mm2=100)


@pytest.fixture
def simple_defects(default_config):
    np.random.seed(7)
    clusters = [
        {"center_x": 60,  "center_y": -40, "n": 35, "spread": 12},
        {"center_x": -80, "center_y": 70,  "n": 25, "spread": 8},
    ]
    return simulate_defects(default_config, n_random=100, clusters=clusters)


@pytest.fixture
def classified_defects(simple_defects):
    return classify_defects(simple_defects, eps=18, min_samples=5)


# ─── WaferConfig ─────────────────────────────────────────────────────────────

class TestWaferConfig:

    def test_default_values(self):
        config = WaferConfig()
        assert config.radius_mm == 150.0
        assert config.edge_exclusion_mm == 3.0
        assert config.die_size_mm2 == 100.0

    def test_custom_values(self):
        config = WaferConfig(radius_mm=100, edge_exclusion_mm=5, die_size_mm2=50)
        assert config.radius_mm == 100
        assert config.edge_exclusion_mm == 5
        assert config.die_size_mm2 == 50


# ─── Defect Simulation ────────────────────────────────────────────────────────

class TestSimulateDefects:

    def test_returns_dataframe(self, default_config):
        df = simulate_defects(default_config, n_random=50)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, default_config):
        df = simulate_defects(default_config, n_random=50)
        for col in ("x", "y", "source"):
            assert col in df.columns

    def test_random_defect_count(self, default_config):
        df = simulate_defects(default_config, n_random=80)
        random_count = len(df[df["source"] == "random"])
        assert random_count == 80

    def test_cluster_defects_added(self, default_config):
        clusters = [{"center_x": 0, "center_y": 0, "n": 20, "spread": 5}]
        df = simulate_defects(default_config, n_random=50, clusters=clusters)
        systematic = df[df["source"] == "systematic"]
        assert len(systematic) == 20

    def test_no_clusters_returns_only_random(self, default_config):
        df = simulate_defects(default_config, n_random=60, clusters=None)
        assert all(df["source"] == "random")

    def test_defects_within_wafer_bounds(self, default_config):
        df = simulate_defects(default_config, n_random=200)
        r_eff = default_config.radius_mm - default_config.edge_exclusion_mm
        distances = np.sqrt(df["x"]**2 + df["y"]**2)
        assert (distances <= r_eff + 1e-6).all()


# ─── Defect Classification ────────────────────────────────────────────────────

class TestClassifyDefects:

    def test_adds_cluster_id_column(self, simple_defects):
        df = classify_defects(simple_defects)
        assert "cluster_id" in df.columns

    def test_adds_classified_as_column(self, simple_defects):
        df = classify_defects(simple_defects)
        assert "classified_as" in df.columns

    def test_classified_as_values(self, simple_defects):
        df = classify_defects(simple_defects)
        assert set(df["classified_as"].unique()).issubset({"systematic", "random"})

    def test_noise_points_classified_as_random(self, simple_defects):
        df = classify_defects(simple_defects)
        noise = df[df["cluster_id"] == -1]
        assert (noise["classified_as"] == "random").all()

    def test_cluster_points_classified_as_systematic(self, simple_defects):
        df = classify_defects(simple_defects)
        clustered = df[df["cluster_id"] >= 0]
        assert (clustered["classified_as"] == "systematic").all()

    def test_tight_clusters_detected(self, default_config):
        """Very tight cluster should always be detected as systematic."""
        np.random.seed(0)
        clusters = [{"center_x": 0, "center_y": 0, "n": 30, "spread": 2}]
        df = simulate_defects(default_config, n_random=20, clusters=clusters)
        df = classify_defects(df, eps=10, min_samples=5)
        systematic = df[df["classified_as"] == "systematic"]
        assert len(systematic) > 0


# ─── Yield Models ─────────────────────────────────────────────────────────────

class TestYieldModels:

    def test_zero_defects_gives_100_percent(self):
        assert poisson_yield(0, 100) == pytest.approx(1.0)
        assert murphy_yield(0, 100) == pytest.approx(1.0)
        assert seeds_yield(0, 100) == pytest.approx(1.0)

    def test_yield_between_0_and_1(self):
        for D0 in [0.001, 0.01, 0.1]:
            assert 0 < poisson_yield(D0, 100) <= 1
            assert 0 < murphy_yield(D0, 100) <= 1
            assert 0 < seeds_yield(D0, 100) <= 1

    def test_higher_defect_density_lower_yield(self):
        assert poisson_yield(0.01, 100) > poisson_yield(0.1, 100)
        assert murphy_yield(0.01, 100) > murphy_yield(0.1, 100)
        assert seeds_yield(0.01, 100) > seeds_yield(0.1, 100)

    def test_larger_die_lower_yield(self):
        D0 = 0.005
        assert poisson_yield(D0, 50) > poisson_yield(D0, 200)
        assert murphy_yield(D0, 50) > murphy_yield(D0, 200)

    def test_murphy_vs_poisson_relationship(self):
        """Murphy yield is generally >= Poisson for same D0 and A."""
        D0, A = 0.005, 100
        assert murphy_yield(D0, A) >= poisson_yield(D0, A) - 1e-6


# ─── Yield Analysis ───────────────────────────────────────────────────────────

class TestComputeYieldAnalysis:

    def test_returns_expected_keys(self, classified_defects, default_config):
        analysis = compute_yield_analysis(classified_defects, default_config)
        for key in ("total_defects", "systematic_defects", "random_defects",
                    "n_clusters", "defect_density_D0", "active_area_mm2",
                    "yield_poisson", "yield_murphy", "yield_seeds",
                    "yield_murphy_random_only"):
            assert key in analysis

    def test_defect_counts_sum_correctly(self, classified_defects, default_config):
        analysis = compute_yield_analysis(classified_defects, default_config)
        assert analysis["systematic_defects"] + analysis["random_defects"] == analysis["total_defects"]

    def test_yield_values_are_percentages(self, classified_defects, default_config):
        analysis = compute_yield_analysis(classified_defects, default_config)
        for key in ("yield_poisson", "yield_murphy", "yield_seeds", "yield_murphy_random_only"):
            assert 0 <= analysis[key] <= 100

    def test_random_only_yield_higher_than_total(self, classified_defects, default_config):
        """Removing systematic defects should always improve yield."""
        analysis = compute_yield_analysis(classified_defects, default_config)
        assert analysis["yield_murphy_random_only"] >= analysis["yield_murphy"]

    def test_defect_density_positive(self, classified_defects, default_config):
        analysis = compute_yield_analysis(classified_defects, default_config)
        assert analysis["defect_density_D0"] > 0

    def test_n_clusters_non_negative(self, classified_defects, default_config):
        analysis = compute_yield_analysis(classified_defects, default_config)
        assert analysis["n_clusters"] >= 0
