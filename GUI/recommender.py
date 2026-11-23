"""Core logic: load data, find category-to-category patterns, rank them."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def load_products(path: Path = DATA_DIR / "products.json") -> pd.DataFrame:
    """Product metadata: asin, title, category, price."""
    return pd.DataFrame(json.loads(Path(path).read_text()))


def load_interactions(path: Path = DATA_DIR / "interactions.json") -> pd.DataFrame:
    """Flatten per-user histories into one row per (user, product) interaction."""
    rows = [
        {"user_id": user["user_id"], **event}
        for user in json.loads(Path(path).read_text())
        for event in user["history"]
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


class Recommender:
    """Ranks the product categories users explore *after* one or more selected categories.

    window: how many later interactions count as "after" (1 = immediate next only).
    """

    def __init__(self, products: pd.DataFrame, interactions: pd.DataFrame, window: int = 2):
        self.window = window
        category_of = dict(zip(products["asin"], products["category"]))
        events = interactions.assign(category=interactions["asin"].map(category_of))
        self.events = events.dropna(subset=["category"])  # already sorted by user, time
        self.categories = sorted(products["category"].unique())
        self.transitions = self._build_transitions()

    def _build_transitions(self) -> pd.DataFrame:
        rows = []
        for user_id, group in self.events.groupby("user_id", sort=False):
            cats, ratings = group["category"].tolist(), group["rating"].tolist()
            for i, source in enumerate(cats):
                for j in range(i + 1, min(i + 1 + self.window, len(cats))):
                    if cats[j] != source:
                        rows.append((user_id, source, cats[j], ratings[j]))
        df = pd.DataFrame(rows, columns=["user_id", "source", "target", "rating"])
        return df.drop_duplicates(["user_id", "source", "target"])

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _as_list(sources) -> list[str]:
        return [sources] if isinstance(sources, str) else list(sources)

    def source_user_count(self, sources) -> int:
        """Distinct users who interacted with any of the source categories."""
        return int(self.events[self.events["category"].isin(self._as_list(sources))]["user_id"].nunique())

    def source_avg_rating(self, sources) -> float:
        return float(self.events[self.events["category"].isin(self._as_list(sources))]["rating"].mean())

    # ------------------------------------------------------------ ranking
    def rank_next(
        self,
        sources,
        min_users: int = 1,
        rating_weight: float = 0.5,
        min_relevance: float = 0.0,
        exclude: set[str] | None = None,
    ) -> pd.DataFrame:
        """Rank next categories after one category, or after any of several combined.

        score     = user_share * ((1 - w) + w * avg_rating / 5)
        relevance = score / sum(scores)   (so the list reads as shares of 100%)
        """
        sources = self._as_list(sources)
        blocked = set(sources) | set(exclude or ())
        subset = self.transitions[self.transitions["source"].isin(sources)]
        subset = subset[~subset["target"].isin(blocked)].drop_duplicates(["user_id", "target"])
        if subset.empty:
            return pd.DataFrame()

        ranked = subset.groupby("target").agg(users=("user_id", "nunique"), avg_rating=("rating", "mean"))
        ranked = ranked[ranked["users"] >= min_users]
        if ranked.empty:
            return pd.DataFrame()

        ranked["user_share"] = ranked["users"] / self.source_user_count(sources)
        ranked["score"] = ranked["user_share"] * ((1 - rating_weight) + rating_weight * ranked["avg_rating"] / 5)
        ranked["relevance"] = ranked["score"] / ranked["score"].sum()
        ranked = ranked[ranked["relevance"] >= min_relevance].sort_values("score", ascending=False)
        return ranked.reset_index().rename(columns={"target": "category"})

    def build_tree(self, sources, levels: int, top_k: int, **rank_kwargs) -> tuple[list[dict], list[dict]]:
        """Expand the top-k next categories level by level (never revisiting a category on the same path)."""
        sources = self._as_list(sources)
        nodes = [{
            "id": "root", "label": " + ".join(sources), "sources": sources, "depth": 0, "parent": None,
            "relevance": 1.0, "path_relevance": 1.0, "users": None, "avg_rating": None,
        }]
        edges: list[dict] = []

        def expand(parent: dict, path: set[str]) -> None:
            if parent["depth"] >= levels:
                return
            recs = self.rank_next(parent["sources"], exclude=path, **rank_kwargs).head(top_k)
            for rec in recs.itertuples():
                node = {
                    "id": f"{parent['id']}>{rec.category}", "label": rec.category, "sources": [rec.category],
                    "depth": parent["depth"] + 1, "parent": parent["id"], "relevance": rec.relevance,
                    "path_relevance": parent["path_relevance"] * rec.relevance,
                    "users": int(rec.users), "avg_rating": float(rec.avg_rating),
                }
                nodes.append(node)
                edges.append({"source": parent["id"], "target": node["id"]})
                expand(node, path | {rec.category})

        expand(nodes[0], set(sources))
        return nodes, edges

    def explain(self, sources, row) -> str:
        sources = self._as_list(sources)
        return (
            f"{int(row.users)} of {self.source_user_count(sources)} customers ({row.user_share:.0%}) "
            f"who viewed {' + '.join(sources)} went on to {row.category} within their next "
            f"{self.window} interaction{'s' if self.window > 1 else ''}, rating it {row.avg_rating:.1f} / 5 on average."
        )
