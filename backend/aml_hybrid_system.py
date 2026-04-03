"""
================================================================================
HYBRID INTELLIGENCE AML RISK SCORING & SAR NARRATIVE GENERATION SYSTEM
================================================================================
Dataset: SAML-D.csv (~1GB, ~9.5M rows)
Target:  Google Colab (16GB RAM), fully offline, explainable, regulator-safe
Author:  Production AML Architecture
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os, gc, time, warnings, json, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix
)

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

# ── Global paths ──────────────────────────────────────────────────────────────
DATA_PATH = "../data/SAML-D.csv"          # update if needed
OUTPUT_DIR  = Path("aml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Chunked ingestion config ──────────────────────────────────────────────────
CHUNK_SIZE  = 250_000               # rows per chunk (~200 MB peak)
MAX_CHUNKS  = 10                 # set to integer for dev/testing (e.g. 10)
RANDOM_SEED = 42

print("=" * 70)
print("AML HYBRID INTELLIGENCE SYSTEM  |  INITIALISING")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0 – MEMORY-SAFE DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
class DataIngestion:
    """
    Chunked CSV loader with dtype optimisation.
    Preserves ALL suspicious rows (Is_laundering == 1) plus a random
    sample of benign rows so downstream layers stay representative.
    """

    DTYPE_MAP = {
        "Sender_account":      "category",
        "Receiver_account":    "category",
        "Payment_currency":    "category",
        "Received_currency":   "category",
        "Sender_bank_location":"category",
        "Receiver_bank_location":"category",
        "Payment_type":        "category",
        "Laundering_type":     "category",
        "Amount":              "float32",
        "Is_laundering":       "int8",
    }

    def __init__(self, filepath: str, chunksize: int = CHUNK_SIZE,
                 max_chunks: int = None, sample_fraction: float = 0.15):
        self.filepath        = filepath
        self.chunksize       = chunksize
        self.max_chunks      = max_chunks
        self.sample_fraction = sample_fraction   # benign row keep-rate

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Stream CSV, keep all suspicious + sampled benign rows."""
        suspicious_chunks, benign_chunks = [], []
        t0 = time.time()

        reader = pd.read_csv(
            self.filepath,
            chunksize=self.chunksize,
            dtype=self.DTYPE_MAP,
            low_memory=False,
            parse_dates=["Date"],
        )

        for i, chunk in enumerate(reader):
            if self.max_chunks and i >= self.max_chunks:
                break

            chunk = self._clean(chunk)

            # Stratified split
            sus = chunk[chunk["Is_laundering"] == 1]
            ben = chunk[chunk["Is_laundering"] == 0].sample(
                frac=self.sample_fraction, random_state=RANDOM_SEED
            )

            if len(sus):   suspicious_chunks.append(sus)
            if len(ben):   benign_chunks.append(ben)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [Chunk {i+1:>4}] suspicious={len(sus):>6,}  "
                      f"benign_sampled={len(ben):>8,}  "
                      f"elapsed={elapsed:.1f}s")
            gc.collect()

        df = pd.concat(suspicious_chunks + benign_chunks, ignore_index=True)
        print(f"\n✓ Loaded {len(df):,} rows  "
              f"(suspicious: {df['Is_laundering'].sum():,}  "
              f"benign: {(df['Is_laundering']==0).sum():,})")
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _clean(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.dropna(subset=["Sender_account", "Receiver_account", "Amount"])
        chunk = chunk[chunk["Amount"] > 0]
        if "Time" in chunk.columns:
            chunk["Time"] = pd.to_datetime(
                chunk["Date"].astype(str) + " " + chunk["Time"].astype(str),
                errors="coerce"
            )
        return chunk


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 – ACCOUNT-LEVEL STATISTICAL FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
class StatisticalFeatureEngineer:
    """Aggregate per Sender_account across the full loaded DataFrame."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[Layer 1] Building statistical features …")
        grp = df.groupby("Sender_account", observed=True)

        feats = pd.DataFrame()

        # Volume / count
        feats["tx_count"]          = grp["Amount"].count()
        feats["unique_receivers"]  = grp["Receiver_account"].nunique()
        feats["unique_currencies"] = grp["Payment_currency"].nunique()

        # Amount statistics
        feats["amt_mean"]          = grp["Amount"].mean().astype("float32")
        feats["amt_std"]           = grp["Amount"].std().fillna(0).astype("float32")
        feats["amt_max"]           = grp["Amount"].max().astype("float32")
        feats["amt_median"]        = grp["Amount"].median().astype("float32")
        feats["amt_cv"]            = (feats["amt_std"] /
                                      (feats["amt_mean"] + 1e-9)).astype("float32")

        # Cross-border ratio
        cross = df[df["Sender_bank_location"] != df["Receiver_bank_location"]]
        cross_cnt = cross.groupby("Sender_account", observed=True)["Amount"].count()
        feats["cross_border_ratio"] = (cross_cnt / feats["tx_count"]).fillna(0).astype("float32")

        # Currency mismatch (payment ≠ received)
        if "Received_currency" in df.columns:
            mismatch = df[df["Payment_currency"] != df["Received_currency"]]
            mis_cnt = mismatch.groupby("Sender_account", observed=True)["Amount"].count()
            feats["currency_mismatch_rate"] = (mis_cnt / feats["tx_count"]).fillna(0).astype("float32")
        else:
            feats["currency_mismatch_rate"] = 0.0

        # Fan-out ratio (unique receivers / tx count)
        feats["fan_out_ratio"]     = (feats["unique_receivers"] /
                                      feats["tx_count"]).astype("float32")

        # Velocity: transactions per day
        if "Time" in df.columns and df["Time"].notna().any():
            date_range = grp["Time"].agg(lambda x: (x.max() - x.min()).days + 1)
            feats["tx_per_day"] = (feats["tx_count"] / date_range.clip(lower=1)).astype("float32")
        else:
            feats["tx_per_day"] = feats["tx_count"].astype("float32")

        # Burst detection: fraction of tx in peak hour (proxy using count variance)
        feats["burst_score"] = feats["amt_cv"].clip(upper=5).astype("float32")

        feats = feats.reset_index().rename(columns={"Sender_account": "account_id"})
        print(f"  ✓ {len(feats):,} accounts  |  {feats.shape[1]} features")
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 – GRAPH CONSTRUCTION & NETWORK FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
class GraphFeatureExtractor:
    """
    Builds a directed weighted transaction graph.
    Extracts structural features: degree, centrality, PageRank, cycle flags.
    For memory safety, uses only the sampled DataFrame (not full 9.5M rows).
    """

    def __init__(self, max_nodes: int = 200_000):
        self.max_nodes = max_nodes
        self.G         = None

    # ------------------------------------------------------------------
    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        print("\n[Layer 2] Building transaction graph …")
        edges = (
            df.groupby(["Sender_account", "Receiver_account"], observed=True)["Amount"]
            .agg(["sum", "count"])
            .reset_index()
        )
        edges.columns = ["src", "dst", "total_amount", "tx_count"]

        # Limit graph size for Colab memory
        top_nodes = set(
            df["Sender_account"].value_counts().head(self.max_nodes).index
        )
        edges = edges[edges["src"].isin(top_nodes) | edges["dst"].isin(top_nodes)]

        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(
                row["src"], row["dst"],
                weight=float(row["total_amount"]),
                count=int(row["tx_count"])
            )

        print(f"  ✓ Graph: {G.number_of_nodes():,} nodes  "
              f"{G.number_of_edges():,} edges")
        self.G = G
        return G

    # ------------------------------------------------------------------
    def extract_features(self, accounts: pd.Index) -> pd.DataFrame:
        G = self.G
        print("  Computing graph metrics …")

        # Degree
        out_deg = dict(G.out_degree())
        in_deg  = dict(G.in_degree())

        # PageRank (mule/funnel detection) – use sparse approx
        print("    PageRank …")
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)

        # Betweenness centrality – approximate for scale
        print("    Betweenness centrality (approx) …")
        bc = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()),
                                        normalized=True, seed=RANDOM_SEED)

        # Cycle detection – nodes in strongly connected components of size > 1
        print("    Cycle detection (SCCs) …")
        scc_map = {}
        for comp in nx.strongly_connected_components(G):
            flag = 1 if len(comp) > 1 else 0
            for node in comp:
                scc_map[node] = flag

        # Build feature frame
        rows = []
        for acct in accounts:
            acct_str = str(acct)
            rows.append({
                "account_id":          acct_str,
                "out_degree":          out_deg.get(acct_str, 0),
                "in_degree":           in_deg.get(acct_str, 0),
                "total_degree":        out_deg.get(acct_str, 0) + in_deg.get(acct_str, 0),
                "pagerank":            pr.get(acct_str, 0.0),
                "betweenness":         bc.get(acct_str, 0.0),
                "in_cycle":            scc_map.get(acct_str, 0),
            })

        gdf = pd.DataFrame(rows)
        print(f"  ✓ Graph features for {len(gdf):,} accounts")
        return gdf


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 – DETERMINISTIC RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class RuleEngine:
    """
    Interpretable AML compliance rules.
    Returns a flag DataFrame + human-readable rule descriptions per account.
    """

    RULES = {
        "R01_high_cross_border":     lambda f: f["cross_border_ratio"]    > 0.80,
        "R02_currency_mismatch":     lambda f: f["currency_mismatch_rate"] > 0.50,
        "R03_high_fan_out":          lambda f: f["out_degree"]             > 50,
        "R04_burst_txns":            lambda f: f["burst_score"]            > 3.0,
        "R05_rapid_velocity":        lambda f: f["tx_per_day"]             > 20,
        "R06_structuring_amounts":   lambda f: (f["amt_mean"] < 10_000) & (f["tx_count"] > 30),
        "R07_round_tripping":        lambda f: f["in_cycle"]               == 1,
        "R08_high_betweenness":      lambda f: f["betweenness"]            > 0.01,
        "R09_many_unique_receivers": lambda f: f["unique_receivers"]       > 40,
        "R10_large_single_txn":      lambda f: f["amt_max"]                > 100_000,
    }

    DESCRIPTIONS = {
        "R01_high_cross_border":     "Cross-border ratio >80%: potential geographic dispersion",
        "R02_currency_mismatch":     "Currency mismatch >50%: potential conversion layering",
        "R03_high_fan_out":          "Out-degree >50: high fan-out structuring pattern",
        "R04_burst_txns":            "Amount CV >3: burst/irregular transaction amounts",
        "R05_rapid_velocity":        "Velocity >20 txns/day: rapid transaction cadence",
        "R06_structuring_amounts":   "Mean amount <$10k & count >30: possible structuring",
        "R07_round_tripping":        "Part of a transaction cycle: round-trip indicator",
        "R08_high_betweenness":      "High betweenness centrality: potential mule role",
        "R09_many_unique_receivers": "Unique receivers >40: scatter/smurfing pattern",
        "R10_large_single_txn":      "Single transaction >$100k: large value movement",
    }

    def apply(self, features: pd.DataFrame) -> pd.DataFrame:
        print("\n[Layer 3] Applying deterministic rule engine …")
        flags = features[["account_id"]].copy()
        for name, rule in self.RULES.items():
            try:
                flags[name] = rule(features).astype("int8")
            except Exception:
                flags[name] = 0
        flags["rule_score"]   = flags[list(self.RULES.keys())].sum(axis=1)
        flags["rule_triggers"] = flags.apply(self._describe, axis=1)
        print(f"  ✓ Rules applied. Accounts with ≥1 flag: "
              f"{(flags['rule_score'] > 0).sum():,}")
        return flags

    def _describe(self, row: pd.Series) -> str:
        fired = [self.DESCRIPTIONS[r] for r in self.RULES if row.get(r, 0) == 1]
        return "; ".join(fired) if fired else "No rules triggered"


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 – UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Isolation Forest on account-level feature matrix.
    Returns normalised anomaly scores [0, 1] where 1 = most anomalous.
    """

    FEATURE_COLS = [
        "tx_count", "unique_receivers", "unique_currencies",
        "amt_mean", "amt_std", "amt_max", "amt_cv",
        "cross_border_ratio", "currency_mismatch_rate",
        "fan_out_ratio", "tx_per_day", "burst_score",
        "out_degree", "in_degree", "total_degree",
        "pagerank", "betweenness", "in_cycle",
    ]

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 max_samples: int = 50_000):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.model         = None
        self.scaler        = StandardScaler()
        self.feature_cols  = None

    # ------------------------------------------------------------------
    def fit_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        print("\n[Layer 4] Isolation Forest anomaly detection …")

        # Use only columns that actually exist
        self.feature_cols = [c for c in self.FEATURE_COLS if c in features.columns]
        X = features[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(X)),
            contamination=self.contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        raw_scores     = self.model.decision_function(X_scaled)  # higher = normal
        anomaly_scores = 1 - (raw_scores - raw_scores.min()) / \
                             (raw_scores.max() - raw_scores.min() + 1e-9)  # [0,1]

        result = features[["account_id"]].copy()
        result["anomaly_score"]    = anomaly_scores.astype("float32")
        result["if_prediction"]    = (self.model.predict(X_scaled) == -1).astype("int8")

        # SHAP-style feature contributions via mean absolute z-score
        result = self._attach_top_features(result, X_scaled, features)

        print(f"  ✓ Anomalies flagged: {result['if_prediction'].sum():,} "
              f"({result['if_prediction'].mean()*100:.1f}%)")
        return result

    # ------------------------------------------------------------------
    def _attach_top_features(self, result, X_scaled, features):
        """Identify top-3 contributing features per account (proxy explainability)."""
        abs_z = np.abs(X_scaled)
        top_idx = np.argsort(abs_z, axis=1)[:, -3:][:, ::-1]
        top_feat_names = []
        top_feat_vals  = []
        for i, row_idx in enumerate(top_idx):
            names = [self.feature_cols[j] for j in row_idx]
            vals  = [float(abs_z[i, j]) for j in row_idx]
            top_feat_names.append(", ".join(names))
            top_feat_vals.append(", ".join(f"{v:.2f}" for v in vals))
        result["top_anomaly_features"]  = top_feat_names
        result["top_anomaly_z_scores"]  = top_feat_vals
        return result


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 – TYPOLOGY MAPPING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class TypologyMapper:
    """
    Maps feature + rule patterns to known AML typologies.
    Uses NO label leakage – purely deterministic from observed signals.
    """

    TYPOLOGIES = {
        "Structuring / Smurfing": {
            "rules": ["R06_structuring_amounts", "R09_many_unique_receivers", "R03_high_fan_out"],
            "graph": lambda g: g["out_degree"] > 30,
            "stat":  lambda s: s["amt_mean"] < 9_000,
        },
        "Layering (Multi-hop)": {
            "rules": ["R01_high_cross_border", "R02_currency_mismatch"],
            "graph": lambda g: g["total_degree"] > 10,
            "stat":  lambda s: s["tx_count"] > 5,
        },
        "Round-Tripping": {
            "rules": ["R07_round_tripping"],
            "graph": lambda g: g["in_cycle"] == 1,
            "stat":  lambda s: s["tx_count"] > 2,
        },
        "Mule Account": {
            "rules": ["R08_high_betweenness"],
            "graph": lambda g: g["betweenness"] > 0.005,
            "stat":  lambda s: s["tx_count"] > 3,
        },
        "Funnel Account": {
            "rules": ["R03_high_fan_out", "R09_many_unique_receivers"],
            "graph": lambda g: g["pagerank"] > 0.001,
            "stat":  lambda s: s["fan_out_ratio"] > 0.5,
        },
        "Rapid Movement / Velocity": {
            "rules": ["R05_rapid_velocity", "R04_burst_txns"],
            "graph": lambda g: g["total_degree"] > 5,
            "stat":  lambda s: s["tx_per_day"] > 15,
        },
    }

    # ------------------------------------------------------------------
    def map(self, merged: pd.DataFrame) -> pd.DataFrame:
        """merged = statistical + graph + rule features joined on account_id."""
        print("\n[Layer 5] Typology mapping …")
        typology_col = []
        confidence   = []

        for _, row in merged.iterrows():
            matched, scores = [], []
            for typo, criteria in self.TYPOLOGIES.items():
                rule_hit  = any(row.get(r, 0) == 1 for r in criteria["rules"])
                try:
                    graph_hit = criteria["graph"](row)
                except Exception:
                    graph_hit = False
                try:
                    stat_hit  = criteria["stat"](row)
                except Exception:
                    stat_hit = False

                score = int(rule_hit) + int(graph_hit) + int(stat_hit)
                if score >= 2:          # require at least 2/3 signals
                    matched.append(typo)
                    scores.append(score)

            if matched:
                best = matched[np.argmax(scores)]
                typology_col.append(best)
                confidence.append(max(scores) / 3)
            else:
                typology_col.append("Unclassified")
                confidence.append(0.0)

        merged["typology"]            = typology_col
        merged["typology_confidence"] = confidence
        cnt = pd.Series(typology_col).value_counts()
        print("  ✓ Typology distribution:\n" +
              "\n".join(f"      {k}: {v}" for k, v in cnt.items()))
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 – CONTEXT CORRELATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class ContextCorrelationEngine:
    """Identify linked accounts and summarise transaction chains for SAR evidence."""

    def __init__(self, G: nx.DiGraph):
        self.G = G

    # ------------------------------------------------------------------
    def build_case(self, account_id: str, df: pd.DataFrame,
                   depth: int = 2) -> dict:
        """Return a structured evidence dict for a single account."""
        G = self.G
        acct = str(account_id)

        # 1-hop and 2-hop neighbourhood
        try:
            successors   = list(nx.ego_graph(G, acct, radius=depth,
                                              undirected=False).nodes())
            predecessors = list(nx.ego_graph(G.reverse(), acct, radius=depth,
                                              undirected=False).nodes())
        except Exception:
            successors, predecessors = [], []

        linked = list(set(successors + predecessors) - {acct})[:30]

        # Suspicious transactions involving this account
        acct_txns = df[
            (df["Sender_account"].astype(str) == acct) |
            (df["Receiver_account"].astype(str) == acct)
        ].head(20)

        tx_summary = []
        for _, t in acct_txns.iterrows():
            tx_summary.append({
                "date":     str(t.get("Date", "N/A")),
                "from":     str(t["Sender_account"]),
                "to":       str(t["Receiver_account"]),
                "amount":   float(t["Amount"]),
                "currency": str(t.get("Payment_currency", "?")),
                "type":     str(t.get("Payment_type", "?")),
            })

        return {
            "account_id":      acct,
            "linked_accounts": linked,
            "tx_chain":        tx_summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 – RAG-BASED SAR NARRATIVE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
class SARNarrativeGenerator:
    """
    Template-based retrieval-augmented generation (in-memory, fully offline).
    Generates regulator-ready SAR narratives strictly grounded in evidence.
    No LLM calls – rules-based composition prevents hallucination.
    """

    TEMPLATES = {
        "Structuring / Smurfing": (
            "ACCOUNT {account_id} conducted {tx_count:,} transactions with a mean value of "
            "${amt_mean:,.2f}, consistently below reporting thresholds, across "
            "{unique_receivers:,} unique counterparties over the review period. "
            "The observed fan-out ratio of {fan_out_ratio:.2f} and geographic dispersion "
            "(cross-border ratio {cross_border_ratio:.1%}) are consistent with structuring "
            "or smurfing activity. The isolation forest anomaly score of {anomaly_score:.3f} "
            "indicates statistically atypical behaviour relative to peer accounts. "
            "Rule triggers: {rule_triggers}."
        ),
        "Layering (Multi-hop)": (
            "ACCOUNT {account_id} exhibited multi-currency layering indicators: "
            "{tx_count:,} transactions with a currency mismatch rate of "
            "{currency_mismatch_rate:.1%} and a cross-border ratio of "
            "{cross_border_ratio:.1%}. Network analysis identified the account as "
            "a node in a {total_degree}-degree transaction chain, suggesting "
            "deliberate placement and layering across jurisdictions. "
            "Anomaly score: {anomaly_score:.3f}. Rule triggers: {rule_triggers}."
        ),
        "Round-Tripping": (
            "ACCOUNT {account_id} is embedded within a detected transaction cycle "
            "(strongly connected component), a classic indicator of round-tripping. "
            "The account processed {tx_count:,} transactions totalling approximately "
            "${amt_max:,.0f} (peak single transaction). The cyclic flow pattern "
            "combined with an anomaly score of {anomaly_score:.3f} warrants "
            "further investigation. Rule triggers: {rule_triggers}."
        ),
        "Mule Account": (
            "ACCOUNT {account_id} demonstrated elevated betweenness centrality "
            "({betweenness:.5f}), positioning it as a critical intermediary node "
            "in the transaction network—a key indicator of a financial mule. "
            "The account received funds from and disbursed to multiple counterparties "
            "({unique_receivers:,} unique receivers). Anomaly score: {anomaly_score:.3f}. "
            "Rule triggers: {rule_triggers}."
        ),
        "Funnel Account": (
            "ACCOUNT {account_id} exhibits high PageRank ({pagerank:.6f}) and "
            "out-degree ({out_degree}), consistent with a funnel account aggregating "
            "and redistributing illicit funds. A fan-out ratio of {fan_out_ratio:.2f} "
            "across {unique_receivers:,} receivers indicates coordinated fund dispersal. "
            "Anomaly score: {anomaly_score:.3f}. Rule triggers: {rule_triggers}."
        ),
        "Rapid Movement / Velocity": (
            "ACCOUNT {account_id} conducted transactions at a velocity of "
            "{tx_per_day:.1f} transactions per day, significantly above peer norms, "
            "with a burst score of {burst_score:.2f}. The rapid movement of funds "
            "across {unique_receivers:,} counterparties is consistent with velocity-based "
            "placement or extraction. Anomaly score: {anomaly_score:.3f}. "
            "Rule triggers: {rule_triggers}."
        ),
        "Unclassified": (
            "ACCOUNT {account_id} was flagged by the AML detection system with an "
            "anomaly score of {anomaly_score:.3f}. While no dominant typology was "
            "conclusively matched, the following rule violations were recorded: "
            "{rule_triggers}. Manual review is recommended to determine SAR eligibility."
        ),
    }

    # ------------------------------------------------------------------
    def generate(self, account_row: pd.Series, context: dict) -> str:
        typology = account_row.get("typology", "Unclassified")
        template = self.TEMPLATES.get(typology, self.TEMPLATES["Unclassified"])

        # Build safe substitution dict from available fields
        fields = account_row.to_dict()
        fields.update({
            "account_id":           str(account_row.get("account_id", "UNKNOWN")),
            "tx_count":             int(fields.get("tx_count", 0)),
            "amt_mean":             float(fields.get("amt_mean", 0)),
            "amt_max":              float(fields.get("amt_max", 0)),
            "unique_receivers":     int(fields.get("unique_receivers", 0)),
            "cross_border_ratio":   float(fields.get("cross_border_ratio", 0)),
            "currency_mismatch_rate": float(fields.get("currency_mismatch_rate", 0)),
            "fan_out_ratio":        float(fields.get("fan_out_ratio", 0)),
            "tx_per_day":           float(fields.get("tx_per_day", 0)),
            "burst_score":          float(fields.get("burst_score", 0)),
            "out_degree":           int(fields.get("out_degree", 0)),
            "total_degree":         int(fields.get("total_degree", 0)),
            "pagerank":             float(fields.get("pagerank", 0)),
            "betweenness":          float(fields.get("betweenness", 0)),
            "anomaly_score":        float(fields.get("anomaly_score", 0)),
            "rule_triggers":        str(fields.get("rule_triggers", "N/A")),
        })

        try:
            narrative = template.format(**fields)
        except (KeyError, ValueError) as e:
            narrative = (
                f"SAR NARRATIVE FOR {fields['account_id']}: "
                f"Anomaly score {fields['anomaly_score']:.3f}, "
                f"typology {typology}. Rule triggers: {fields['rule_triggers']}. "
                f"[Template error: {e}]"
            )

        # Append linked accounts if present
        if context.get("linked_accounts"):
            linked_str = ", ".join(str(a) for a in context["linked_accounts"][:5])
            narrative += f" Linked accounts identified: {linked_str}."

        return narrative


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 8 – AUDIT TRAIL & EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
class AuditTrail:
    """Persist per-account evidence packages as JSON for regulatory review."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.records    = []

    # ------------------------------------------------------------------
    def log(self, account_row: pd.Series, context: dict, narrative: str):
        record = {
            "timestamp":          datetime.utcnow().isoformat() + "Z",
            "account_id":         str(account_row.get("account_id")),
            "anomaly_score":      float(account_row.get("anomaly_score", 0)),
            "rule_score":         int(account_row.get("rule_score", 0)),
            "typology":           str(account_row.get("typology", "?")),
            "typology_confidence":float(account_row.get("typology_confidence", 0)),
            "top_anomaly_features":str(account_row.get("top_anomaly_features", "")),
            "rule_triggers":      str(account_row.get("rule_triggers", "")),
            "statistical_features": {
                k: round(float(account_row[k]), 4)
                for k in ["tx_count", "amt_mean", "amt_std", "amt_max",
                          "cross_border_ratio", "currency_mismatch_rate",
                          "fan_out_ratio", "tx_per_day"]
                if k in account_row.index
            },
            "graph_features": {
                k: round(float(account_row[k]), 6)
                for k in ["out_degree", "in_degree", "pagerank",
                          "betweenness", "in_cycle"]
                if k in account_row.index
            },
            "linked_accounts":    context.get("linked_accounts", [])[:10],
            "tx_chain_sample":    context.get("tx_chain", [])[:5],
            "sar_narrative":      narrative,
        }
        self.records.append(record)

    # ------------------------------------------------------------------
    def save(self):
        path = self.output_dir / "audit_trail.json"
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
        print(f"\n✓ Audit trail saved: {path}  ({len(self.records):,} records)")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Evaluator:
    """Account-level evaluation using Is_laundering ground truth."""

    def evaluate(self, merged: pd.DataFrame, df_raw: pd.DataFrame,
                 output_dir: Path) -> dict:
        print("\n[Evaluation] Computing account-level metrics …")

        # Aggregate ground truth at account level
        gt = (
            df_raw.groupby("Sender_account", observed=True)["Is_laundering"]
            .max()
            .reset_index()
            .rename(columns={"Sender_account": "account_id",
                              "Is_laundering":  "true_label"})
        )
        gt["account_id"] = gt["account_id"].astype(str)

        eval_df = merged.merge(gt, on="account_id", how="left")
        eval_df["true_label"] = eval_df["true_label"].fillna(0).astype(int)

        y_true   = eval_df["true_label"].values
        y_score  = eval_df["anomaly_score"].values
        y_pred   = eval_df["if_prediction"].values

        # PR-AUC
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
        prauc = auc(rec_arr, prec_arr)

        # Standard metrics at IF threshold
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)

        # Recall @ top 1%
        k        = max(1, int(len(eval_df) * 0.01))
        top_k    = eval_df.nlargest(k, "anomaly_score")["true_label"]
        recall_k = top_k.sum() / max(1, y_true.sum())

        # FPR
        cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        fpr = fp / max(1, tn + fp)

        metrics = {
            "PR_AUC":           round(prauc, 4),
            "Precision":        round(prec, 4),
            "Recall":           round(rec, 4),
            "F1_Score":         round(f1, 4),
            "Recall_at_Top1pct":round(recall_k, 4),
            "False_Positive_Rate": round(fpr, 4),
            "Total_Accounts":   int(len(eval_df)),
            "True_Suspicious":  int(y_true.sum()),
            "Flagged":          int(y_pred.sum()),
        }

        print("  ── EVALUATION RESULTS ───────────────────────────")
        for k2, v in metrics.items():
            print(f"     {k2:<25}: {v}")

        # Save
        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._plot_pr_curve(prec_arr, rec_arr, prauc, output_dir)
        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _plot_pr_curve(prec, rec, prauc, output_dir):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rec, prec, lw=2, color="#1f77b4",
                label=f"PR-AUC = {prauc:.4f}")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve (Account Level)", fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curve.png", dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
class VisualisationDashboard:

    def plot_all(self, merged: pd.DataFrame, output_dir: Path):
        print("\n[Visualisation] Generating diagnostic plots …")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("AML Hybrid System – Diagnostic Dashboard", fontsize=14, y=1.01)

        # 1. Anomaly score distribution
        ax = axes[0, 0]
        merged["anomaly_score"].hist(bins=60, ax=ax, color="#d62728", alpha=0.7)
        ax.set_title("Anomaly Score Distribution")
        ax.set_xlabel("Score"); ax.set_ylabel("Count")

        # 2. Rule score histogram
        ax = axes[0, 1]
        merged["rule_score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#1f77b4")
        ax.set_title("Rule Score Distribution")
        ax.set_xlabel("Rules Triggered"); ax.set_ylabel("Accounts")

        # 3. Typology breakdown
        ax = axes[0, 2]
        typ_cnt = merged["typology"].value_counts()
        typ_cnt.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Typology Mapping")
        ax.set_xlabel("Accounts")

        # 4. Anomaly score vs rule score scatter
        ax = axes[1, 0]
        ax.scatter(merged["rule_score"], merged["anomaly_score"],
                   alpha=0.3, s=5, c="#2ca02c")
        ax.set_xlabel("Rule Score"); ax.set_ylabel("Anomaly Score")
        ax.set_title("Rules vs Anomaly Score")

        # 5. Cross-border ratio distribution
        ax = axes[1, 1]
        merged["cross_border_ratio"].hist(bins=50, ax=ax, color="#9467bd", alpha=0.7)
        ax.set_title("Cross-Border Ratio Distribution")
        ax.set_xlabel("Ratio")

        # 6. PageRank distribution (log scale)
        ax = axes[1, 2]
        pr_vals = merged["pagerank"].clip(lower=1e-8)
        ax.hist(np.log10(pr_vals + 1e-8), bins=60, color="#ff7f0e", alpha=0.7)
        ax.set_title("PageRank Distribution (log10)")
        ax.set_xlabel("log10(PageRank)")

        fig.tight_layout()
        fig.savefig(output_dir / "dashboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Dashboard saved: {output_dir / 'dashboard.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
class AMLPipeline:

    def __init__(self, data_path: str = DATA_PATH,
                 output_dir: Path = OUTPUT_DIR):
        self.data_path  = data_path
        self.output_dir = output_dir
        self.df         = None
        self.merged     = None

    # ------------------------------------------------------------------
    def run(self, top_k_sar: int = 50):
        t_total = time.time()
        print("\n" + "=" * 70)
        print("STARTING HYBRID AML PIPELINE")
        print("=" * 70)

        # ── Layer 0: Ingest ──────────────────────────────────────────
        ingestion = DataIngestion(self.data_path, CHUNK_SIZE, MAX_CHUNKS)
        self.df   = ingestion.load()

        # ── Layer 1: Statistical features ───────────────────────────
        stat_eng  = StatisticalFeatureEngineer()
        stat_feat = stat_eng.build(self.df)

        # ── Layer 2: Graph features ──────────────────────────────────
        gfx = GraphFeatureExtractor(max_nodes=150_000)
        G   = gfx.build_graph(self.df)
        grph_feat = gfx.extract_features(stat_feat["account_id"])

        # ── Merge stat + graph ───────────────────────────────────────
        merged = stat_feat.merge(grph_feat, on="account_id", how="left")
        for col in ["out_degree", "in_degree", "total_degree", "in_cycle"]:
            merged[col] = merged.get(col, 0).fillna(0).astype(int)
        for col in ["pagerank", "betweenness"]:
            merged[col] = merged.get(col, 0.0).fillna(0.0).astype("float32")

        # ── Layer 3: Rules ───────────────────────────────────────────
        rule_eng  = RuleEngine()
        rule_feat = rule_eng.apply(merged)
        merged    = merged.merge(rule_feat, on="account_id", how="left")

        # ── Layer 4: Isolation Forest ────────────────────────────────
        detector  = AnomalyDetector()
        anom_feat = detector.fit_predict(merged)
        merged    = merged.merge(anom_feat, on="account_id", how="left")

        # ── Layer 5: Typology mapping ────────────────────────────────
        typo_mapper = TypologyMapper()
        merged      = typo_mapper.map(merged)

        self.merged = merged

        # ── Layer 6 + 7 + 8: SAR cases for top-K anomalous accounts ─
        print(f"\n[Layer 6-8] Building SAR cases for top {top_k_sar} accounts …")
        top_accounts = merged.nlargest(top_k_sar, "anomaly_score")
        sar_gen    = SARNarrativeGenerator()
        ctx_engine = ContextCorrelationEngine(G)
        audit      = AuditTrail(self.output_dir)

        sar_records = []
        for _, row in top_accounts.iterrows():
            context   = ctx_engine.build_case(row["account_id"], self.df)
            narrative = sar_gen.generate(row, context)
            audit.log(row, context, narrative)
            sar_records.append({
                "account_id":    row["account_id"],
                "anomaly_score": row["anomaly_score"],
                "typology":      row["typology"],
                "narrative":     narrative,
            })

        audit.save()

        # Save SAR narratives as CSV
        sar_df = pd.DataFrame(sar_records)
        sar_path = self.output_dir / "sar_narratives.csv"
        sar_df.to_csv(sar_path, index=False)
        print(f"  ✓ SAR narratives saved: {sar_path}")

        # Save full risk scores
        risk_path = self.output_dir / "account_risk_scores.csv"
        merged.to_csv(risk_path, index=False)
        print(f"  ✓ Risk scores saved:    {risk_path}")

        # ── Evaluation ───────────────────────────────────────────────
        evaluator = Evaluator()
        metrics   = evaluator.evaluate(merged, self.df, self.output_dir)

        # ── Visualisation ────────────────────────────────────────────
        viz = VisualisationDashboard()
        viz.plot_all(merged, self.output_dir)

        elapsed = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"✅  PIPELINE COMPLETE  |  Total time: {elapsed:.1f}s")
        print(f"   Outputs in: {self.output_dir.resolve()}")
        print(f"{'='*70}\n")

        return merged, metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = AMLPipeline(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
    )
    results, metrics = pipeline.run(top_k_sar=100)

    # Quick preview of top flagged accounts
    print("\nTop 10 highest-risk accounts:")
    preview_cols = ["account_id", "anomaly_score", "rule_score",
                    "typology", "tx_count", "amt_mean"]
    available = [c for c in preview_cols if c in results.columns]
    print(results.nlargest(10, "anomaly_score")[available].to_string(index=False))
