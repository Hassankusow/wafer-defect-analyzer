"""
Wafer Defect Clustering & Yield Impact Analyzer

Classifies defects as systematic (process-induced) vs. random (Poisson-distributed)
using DBSCAN spatial clustering and models die yield using Murphy and Poisson models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from dataclasses import dataclass


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class WaferConfig:
    radius_mm: float = 150.0       # 300mm wafer
    edge_exclusion_mm: float = 3.0
    die_size_mm2: float = 100.0    # e.g., 10mm x 10mm die


# ─── Defect Generation (Simulation) ──────────────────────────────────────────

def simulate_defects(config: WaferConfig,
                     n_random: int = 120,
                     clusters: list = None) -> pd.DataFrame:
    """
    Simulate wafer defect map with random (Poisson) and systematic (clustered) defects.
    clusters: list of dicts with keys center_x, center_y, n, spread
    """
    r_eff = config.radius_mm - config.edge_exclusion_mm

    # Random Poisson-distributed defects
    theta = np.random.uniform(0, 2 * np.pi, n_random * 3)
    r = r_eff * np.sqrt(np.random.uniform(0, 1, n_random * 3))
    x_all = r * np.cos(theta)
    y_all = r * np.sin(theta)
    mask = (x_all**2 + y_all**2) <= r_eff**2
    x_rand = x_all[mask][:n_random]
    y_rand = y_all[mask][:n_random]

    records = [{"x": x, "y": y, "source": "random"} for x, y in zip(x_rand, y_rand)]

    # Systematic clustered defects
    if clusters:
        for cl in clusters:
            cx, cy = cl["center_x"], cl["center_y"]
            n, spread = cl["n"], cl["spread"]
            x_cl = np.random.normal(cx, spread, n)
            y_cl = np.random.normal(cy, spread, n)
            for x, y in zip(x_cl, y_cl):
                records.append({"x": x, "y": y, "source": "systematic"})

    return pd.DataFrame(records)


# ─── Clustering ───────────────────────────────────────────────────────────────

def classify_defects(df: pd.DataFrame,
                     eps: float = 15.0,
                     min_samples: int = 5) -> pd.DataFrame:
    """
    Apply DBSCAN to identify systematic defect clusters.
    Noise points (label=-1) are treated as random/Poisson-distributed.
    """
    coords = df[["x", "y"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df = df.copy()
    df["cluster_id"] = db.labels_
    df["classified_as"] = df["cluster_id"].apply(
        lambda c: "systematic" if c >= 0 else "random"
    )
    return df


# ─── Yield Models ────────────────────────────────────────────────────────────

def poisson_yield(D0: float, A: float) -> float:
    """Poisson yield model: Y = exp(-D0 * A)"""
    return np.exp(-D0 * A)


def murphy_yield(D0: float, A: float) -> float:
    """Murphy yield model: Y = ((1 - exp(-D0*A)) / (D0*A))^2"""
    if D0 * A < 1e-9:
        return 1.0
    return ((1 - np.exp(-D0 * A)) / (D0 * A)) ** 2


def seeds_yield(D0: float, A: float, alpha: float = 3.0) -> float:
    """Seeds (negative binomial) yield model: Y = (1 + D0*A/alpha)^(-alpha)"""
    return (1 + D0 * A / alpha) ** (-alpha)


def compute_yield_analysis(df: pd.DataFrame, config: WaferConfig) -> dict:
    """Compute defect density and yield predictions across all three models."""
    r_eff = config.radius_mm - config.edge_exclusion_mm
    active_area_mm2 = np.pi * r_eff ** 2
    A = config.die_size_mm2

    total = len(df)
    systematic = len(df[df["classified_as"] == "systematic"])
    random_count = len(df[df["classified_as"] == "random"])

    D0_total = total / active_area_mm2
    D0_random = random_count / active_area_mm2  # Poisson component only

    return {
        "total_defects": total,
        "systematic_defects": systematic,
        "random_defects": random_count,
        "n_clusters": df["cluster_id"].nunique() - (1 if -1 in df["cluster_id"].values else 0),
        "defect_density_D0": round(D0_total, 6),
        "active_area_mm2": round(active_area_mm2, 2),
        "die_size_mm2": A,
        "yield_poisson":  round(poisson_yield(D0_total, A) * 100, 2),
        "yield_murphy":   round(murphy_yield(D0_total, A) * 100, 2),
        "yield_seeds":    round(seeds_yield(D0_total, A) * 100, 2),
        "yield_murphy_random_only": round(murphy_yield(D0_random, A) * 100, 2),
    }


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_wafer_map(df: pd.DataFrame, analysis: dict,
                   config: WaferConfig, save_path: str = None):
    """Render wafer defect map with cluster overlays and yield annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, color_by in zip(axes, ["classified_as", "cluster_id"]):
        # Wafer boundary
        wafer = plt.Circle((0, 0), config.radius_mm, fill=False,
                            color="black", linewidth=2)
        edge = plt.Circle((0, 0), config.radius_mm - config.edge_exclusion_mm,
                           fill=False, color="gray", linestyle="--", linewidth=1)
        ax.add_patch(wafer)
        ax.add_patch(edge)

        if color_by == "classified_as":
            colors = {"random": "royalblue", "systematic": "crimson"}
            for label, grp in df.groupby("classified_as"):
                ax.scatter(grp["x"], grp["y"], c=colors[label], s=8,
                           alpha=0.6, label=label)
            ax.set_title("Defect Classification\n(blue=random, red=systematic)")
        else:
            noise = df[df["cluster_id"] == -1]
            clustered = df[df["cluster_id"] >= 0]
            ax.scatter(noise["x"], noise["y"], c="royalblue", s=6,
                       alpha=0.4, label="noise")
            if not clustered.empty:
                sc = ax.scatter(clustered["x"], clustered["y"],
                                c=clustered["cluster_id"], cmap="tab10",
                                s=12, alpha=0.9)
                plt.colorbar(sc, ax=ax, label="Cluster ID")
            ax.set_title("DBSCAN Cluster Labels")

        lim = config.radius_mm + 15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    fig.suptitle(
        f"Wafer Defect Analysis  |  D0={analysis['defect_density_D0']:.5f} def/mm²  |  "
        f"Murphy Yield: {analysis['yield_murphy']}%  |  "
        f"Clusters: {analysis['n_clusters']}",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(7)

    config = WaferConfig(radius_mm=150, edge_exclusion_mm=3, die_size_mm2=100)

    # Simulate: 100 random + 2 systematic clusters (e.g., scratch + hotspot)
    clusters = [
        {"center_x": 60,  "center_y": -40, "n": 35, "spread": 12},
        {"center_x": -80, "center_y": 70,  "n": 25, "spread": 8},
    ]
    df = simulate_defects(config, n_random=100, clusters=clusters)
    df = classify_defects(df, eps=18, min_samples=5)

    analysis = compute_yield_analysis(df, config)

    print("=== Wafer Yield Analysis ===")
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    plot_wafer_map(df, analysis, config, save_path="wafer_defect_map.png")
