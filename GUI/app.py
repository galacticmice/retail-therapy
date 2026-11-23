"""Amazon Product Relevance - category-to-category front-end prototype.

Run with:  streamlit run app.py
"""
import inspect
from collections import defaultdict

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from recommender import Recommender, load_interactions, load_products

st.set_page_config(page_title="Product Relevance", page_icon="🛒", layout="wide")

DARK, DARK2, ORANGE, TEAL = "#3c4147", "#555b62", "#ff9900", "#007185"
LEVEL_COLORS = [DARK2, ORANGE, TEAL, "#7d8c9c"]
LEVEL_NAMES = ["You selected", "Next step", "Step after that", "Third step"]

CSS = f"""
<style>
/* Everything is sized in rem, and the root size follows the window width, so boxes, text and
   headlines grow and shrink together (and zooming out makes the page fill the screen). */
html {{font-size: clamp(15px, 0.85vw, 44px) !important;}}
.block-container {{padding: 1.2rem 2rem 3rem 2rem; max-width: none !important;}}
[data-testid="stSidebar"][aria-expanded="true"] {{min-width: 21rem !important; max-width: 21rem !important; width: 21rem !important;}}
#MainMenu, footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{background: transparent;}}
h1 a, h2 a, h3 a {{display: none;}}
[data-testid="stVerticalBlockBorderWrapper"] {{background: #fff; border: 1px solid #d5d9d9; border-radius: 0.5rem;}}
[data-baseweb="tag"] {{background: {DARK2};}}
[data-baseweb="tag"] span {{color: #fff;}}
[data-baseweb="tab-list"] {{gap: 1.4rem;}}
[data-testid="stMain"] [data-testid="stButtonGroup"] {{justify-content: flex-end;}}
[data-testid="stForm"] {{border: none; padding: 0;}}
[data-testid="stForm"] button, [data-testid="stForm"] button:hover {{background: #ffd814 !important; border: 1px solid #fcd200 !important;
    color: #2e3338 !important; font-weight: 600; border-radius: 0.5rem;}}
[data-testid="stForm"] button:hover {{background: #f7ca00 !important;}}
.az-header {{background: {DARK}; border-bottom: 0.2rem solid {ORANGE}; border-radius: 0.5rem 0.5rem 0 0; padding: 0.9rem 1.2rem;}}
.az-header span {{color: #fff; font-size: 1.4rem; font-weight: 700; letter-spacing: -0.01em;}}
.az-header em {{color: {ORANGE}; font-style: normal;}}
.az-nav {{background: {DARK2}; color: #e3e6e6; font-size: 0.9rem; padding: 0.45rem 1.2rem;
          border-radius: 0 0 0.5rem 0.5rem; margin-bottom: 1rem;}}
.sec-title {{font-size: 1.25rem; font-weight: 700; margin: 0.3rem 0 0.1rem 0;}}
.sec-title b {{color: {TEAL};}}
.sec-meta {{color: #565959; font-size: 0.9rem; margin: 0 0 0.8rem 0;}}
.az-card {{border: 1px solid #d5d9d9; border-radius: 0.5rem; padding: 0.8rem 0.9rem; background: #fff; height: 100%;}}
.az-rank {{display: inline-block; background: {ORANGE}; color: {DARK}; font-weight: 700; font-size: 0.78rem;
           border-radius: 0.25rem; padding: 0.05rem 0.45rem; margin-bottom: 0.4rem;}}
.az-cat {{color: {TEAL}; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.15rem;}}
.az-stars {{color: #ffa41c; font-size: 1rem; letter-spacing: 0.06rem;}}
.az-stars span {{color: #565959; font-size: 0.85rem; letter-spacing: 0; margin-left: 0.3rem;}}
.az-stat {{font-size: 0.88rem; margin: 0.5rem 0 0.4rem 0;}}
.az-bar {{background: #e7e7e7; border-radius: 0.25rem; height: 0.4rem;}}
.az-bar div {{background: {ORANGE}; height: 0.4rem; border-radius: 0.25rem;}}
.az-rel {{color: #565959; font-size: 0.8rem; margin-top: 0.25rem;}}
.help-note {{color: #565959; font-size: 0.9rem; margin: 0 0 0.6rem 0;}}
/* chart text scales with the page too */
.js-plotly-plot .main-svg text {{font-size: 0.8rem !important;}}
.js-plotly-plot .main-svg .gtitle {{font-size: 1rem !important;}}
</style>
"""

# Works on both older and newer Streamlit versions (the full-width argument was renamed).
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_DF_HAS_WIDTH = "width" in inspect.signature(st.dataframe).parameters
DF_FULL_WIDTH = {"width": "stretch"} if _DF_HAS_WIDTH else {"use_container_width": True}


def show_plot(fig: go.Figure, **kwargs) -> None:
    full_width = {"width": "stretch"} if _PLOTLY_HAS_WIDTH else {"use_container_width": True}
    st.plotly_chart(fig, **full_width, **kwargs)


def choice(label, options, default, key, format_func=str, help=None, collapse=False):
    """Button-style single choice (segmented control; falls back to radio on older Streamlit)."""
    memory = f"_last_{key}"
    st.session_state.setdefault(memory, default)
    visibility = "collapsed" if collapse else "visible"
    if hasattr(st, "segmented_control"):
        value = st.segmented_control(label, options, default=default, key=key, format_func=format_func,
                                     help=help, label_visibility=visibility)
    else:
        value = st.radio(label, options, index=options.index(default), horizontal=True, key=key,
                         format_func=format_func, help=help, label_visibility=visibility)
    if value is None:  # the user clicked the active button again: keep the previous choice
        return st.session_state[memory]
    st.session_state[memory] = value
    return value


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"size": 12}, title_font={"size": 14})
    return fig


@st.cache_data
def load_data():
    return load_products(), load_interactions()


@st.cache_resource
def get_recommender(window: int) -> Recommender:
    products, interactions = load_data()
    return Recommender(products, interactions, window)


def layout_tree(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    """Root on the left, children to the right; each parent sits at the middle of its children."""
    children = defaultdict(list)
    for n in nodes:
        if n["parent"]:
            children[n["parent"]].append(n["id"])
    depth = {n["id"]: n["depth"] for n in nodes}
    pos, next_row = {}, [0]

    def place(node_id: str) -> float:
        kids = children[node_id]
        if kids:
            ys = [place(k) for k in kids]
            y = sum(ys) / len(ys)
        else:
            y =next_row[0]
            next_row[0] += 1
        pos[node_id] = (depth[node_id], -y)
        return y

    place("root")
    return pos


# Hover boxes: non-breaking spaces and blank first/last lines give the text some padding.
_PAD = "\u00a0\u00a0\u00a0"
HOVER_NODE = "<br>".join(["\u00a0"] + [_PAD + line + _PAD for line in [
    "<b>%{customdata[0]}</b>",
    "Relevance after parent: %{customdata[1]:.1%}",
    "Path relevance: %{customdata[2]:.1%}",
    "Customers: %{customdata[3]}",
    "Avg rating: %{customdata[4]:.2f}",
]] + ["\u00a0"]) + "<extra></extra>"
HOVER_ROOT = "<br>".join(["\u00a0"] + [_PAD + line + _PAD for line in ["<b>%{customdata[0]}</b>", "Your selection"]]
                         + ["\u00a0"]) + "<extra></extra>"


def tree_figure(nodes: list[dict], edges: list[dict], levels: int) -> go.Figure:
    pos = layout_tree(nodes)
    edge_x, edge_y = [], []
    for e in edges:
        (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", showlegend=False,
                               line={"width": 1.5, "color": "rgba(125,140,156,0.6)"}))

    for depth in sorted({n["depth"] for n in nodes}):
        group = [n for n in nodes if n["depth"] == depth]
        fig.add_trace(go.Scatter(
            x=[pos[n["id"]][0] for n in group],
            y=[pos[n["id"]][1] for n in group],
            mode="markers+text",
            name=LEVEL_NAMES[depth],
            text=[n["label"] if depth == 0 else f"{n['label']}<br>{n['relevance']:.0%}" for n in group],
            textposition="top center",
            textfont={"size": 11},
            customdata=[[n["label"], n["relevance"], n["path_relevance"], n["users"], n["avg_rating"]]
                        for n in group],
            hovertemplate=HOVER_ROOT if depth == 0 else HOVER_NODE,
            marker={
                "size": [38 if depth == 0 else 16 + 46 * n["relevance"] for n in group],
                "color": LEVEL_COLORS[depth],
                "line": {"width": 2, "color": "white"},
            },
        ))

    rows = sum(1 for n in nodes if not any(e["source"] == n["id"] for e in edges))
    fig.update_layout(
        height=max(380, 85 * rows + 120),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False, "range": [-0.5, levels + 0.5]},
        yaxis={"visible": False},
        hovermode="closest",
        hoverlabel={"bgcolor": "white", "bordercolor": "#d5d9d9", "align": "left",
                    "font": {"color": "#3a3f44", "size": 13}},
        legend={"orientation": "h", "y": -0.03},
    )
    return style_fig(fig)


def card_html(rank: int, row, max_relevance: float) -> str:
    filled = round(row.avg_rating)
    stars = "★" * filled + "☆" * (5 - filled)
    return (
        f'<div class="az-card"><span class="az-rank">#{rank}</span>'
        f'<div class="az-cat">{row.category}</div>'
        f'<div class="az-stars">{stars}<span>{row.avg_rating:.1f}</span></div>'
        f'<div class="az-stat"><b>{row.user_share:.0%}</b> of customers went next</div>'
        f'<div class="az-bar"><div style="width:{row.relevance / max_relevance * 100:.0f}%"></div></div>'
        f'<div class="az-rel">{row.relevance:.0%} relevance</div></div>'
    )


# sidebar settings
MIN_CUSTOMERS = {"Any": 1, "20+": 20, "50+": 50, "100+": 100}
RATING_INFLUENCE = {"Popularity": 0.0, "Balanced": 0.5, "Ratings": 1.0}

st.markdown(CSS, unsafe_allow_html=True)
with st.sidebar:
    st.subheader("Settings")
    top_k = choice("Categories per level", [1, 2, 3, 4, 5], 4, key="top_k")
    levels = st.slider("Recommendation levels", 1, 3, 2)
    with st.expander("Fine-tune the ranking", expanded=True):
        window = choice("Look-ahead window", [1, 2, 3, 4, 5], 2, key="window",
                        help="A category counts as 'next' if it appears within this many of the customer's "
                             "following interactions. 1 = only the very next one. 2 = either of the next two, and so on.")
        st.caption("Counts a category as next if it appears in the customer's "
                   + ("very next interaction." if window == 1 else f"next {window} interactions."))
        min_users = MIN_CUSTOMERS[choice("Minimum supporting customers", list(MIN_CUSTOMERS), "20+", key="min_users",
                                         help="Hide categories backed by fewer customers than this.")]
        rating_weight = RATING_INFLUENCE[choice(
            "Rating influence", list(RATING_INFLUENCE), "Balanced", key="rating",
            help="Popularity ranks only by how many customers went next. Balanced blends in the average rating. "
                 "Ratings counts the average rating fully.")]

rec = get_recommender(window)
products, interactions = load_data()

# header + search
st.markdown('<div class="az-header"><span>Amazon <em>Product Relevance</em></span></div>'
            '<div class="az-nav">See which categories customers explore after the ones they are viewing.</div>',
            unsafe_allow_html=True)

with st.container(border=True):
    with st.form("search_form"):
        selected = st.multiselect("Search categories", rec.categories, default=["Laptop"],
                                  placeholder="Type to search, for example Laptop or Webcam. Add as many as you like.")
        st.form_submit_button("Search")
    mode = "Show each separately"
    if len(selected) > 1:
        mode = st.radio("How should multiple categories be shown?", ["Show each separately", "Combine into one list"],
                        horizontal=True,
                        help="Combine ranks what customers explore after viewing any of the selected categories.")

if not selected:
    st.info("Pick at least one category above, then press Search.")
    st.stop()

groups = [[c] for c in selected] if mode == "Show each separately" else [selected]
rank_kwargs = {"min_users": min_users, "rating_weight": rating_weight}
results = [(sources, " + ".join(sources), rec.rank_next(sources, **rank_kwargs)) for sources in groups]

st.write("")
tab_next, tab_tree, tab_data = st.tabs(["Summary", "Probability Tree", "Data Overview"])

VIEW_ICONS = {"boxes": "▦", "list": "☰"}

# summary (one format at a time)
with tab_next:
    for i, (sources, label, ranked) in enumerate(results):
        with st.container(border=True):
            title_col, toggle_col = st.columns([8, 1])
            title_col.markdown(f'<p class="sec-title">Customers who viewed <b>{label}</b> also explore</p>',
                               unsafe_allow_html=True)
            if ranked.empty:
                st.warning("No categories meet these settings. Try a lower minimum supporting customers.")
                continue
            with toggle_col:
                view = choice("View", list(VIEW_ICONS), "boxes", key=f"view-{i}",
                              format_func=VIEW_ICONS.get, collapse=True)
            st.markdown(
                f'<p class="sec-meta">Based on {rec.source_user_count(sources):,} customers who rated these products '
                f'{rec.source_avg_rating(sources):.1f} / 5 on average. {len(ranked)} categories follow.</p>',
                unsafe_allow_html=True)

            if view == "boxes":
                top = ranked.head(top_k)
                max_rel = top["relevance"].max()
                for col, (n, row) in zip(st.columns(len(top)), enumerate(top.itertuples(), start=1)):
                    col.markdown(card_html(n, row, max_rel), unsafe_allow_html=True)
                if len(ranked) > len(top):
                    st.caption(f"Showing the top {len(top)} of {len(ranked)} categories. Switch to the list view to see all of them.")
            else:
                table = ranked.assign(
                    Relevance=ranked["relevance"] * 100,
                    **{"Customers who went next": ranked["users"], "Share of customers": ranked["user_share"] * 100},
                ).rename(columns={"category": "Category", "avg_rating": "Avg rating"})
                st.dataframe(
                    table[["Category", "Relevance", "Customers who went next", "Share of customers", "Avg rating"]],
                    hide_index=True,
                    **DF_FULL_WIDTH,
                    column_config={
                        "Relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=100, format="%.1f%%"),
                        "Share of customers": st.column_config.NumberColumn(format="%.0f%%"),
                        "Avg rating": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                st.caption(f"Score = share of customers × ({1 - rating_weight:.0%} popularity + "
                           f"{rating_weight:.0%} average rating ÷ 5). Relevance is each score as a share of the total.")

            best = next(ranked.itertuples())
            st.markdown(f'<p class="help-note" style="margin-top:0.8rem">{rec.explain(sources, best)}</p>',
                        unsafe_allow_html=True)

# tree
with tab_tree:
    st.markdown('<p class="help-note">Follow a path from left to right. Larger circles mean higher relevance. '
                'Hover a circle for customers and ratings.</p>', unsafe_allow_html=True)
    for i, (sources, label, ranked) in enumerate(results):
        with st.container(border=True):
            st.markdown(f'<p class="sec-title">Probability tree from <b>{label}</b></p>', unsafe_allow_html=True)
            if ranked.empty:
                st.warning("No categories meet these settings.")
                continue
            nodes, edges = rec.build_tree(sources, levels, top_k, **rank_kwargs)
            show_plot(tree_figure(nodes, edges, levels), key=f"tree-{i}")

#dataset overview
with tab_data:
    d1, d2, d3 = st.columns(3)
    for col, name, value in [(d1, "Categories", len(rec.categories)),
                             (d2, "Customers", interactions["user_id"].nunique()),
                             (d3, "Interactions", len(interactions))]:
        with col, st.container(border=True):
            st.metric(name, f"{value:,}")

    left, right = st.columns(2)
    hist = px.histogram(interactions, x="rating", nbins=5, title="Rating distribution", height=300)
    hist.update_traces(marker_color=ORANGE)
    per_cat = rec.events.groupby("category").size().sort_values().reset_index(name="Interactions")
    bars = px.bar(per_cat, x="Interactions", y="category", orientation="h", title="Interactions by category", height=300)
    bars.update_traces(marker_color=ORANGE)
    bars.update_layout(yaxis_title=None)
    with left, st.container(border=True):
        show_plot(style_fig(hist), key="hist")
    with right, st.container(border=True):
        show_plot(style_fig(bars), key="bars")

    st.caption("Prototype note: this uses generated sample data. Swap in the real files in data/ "
               "(see README.md) and the same views will work. Pandas and Seaborn analysis can build on "
               "`recommender.Recommender.transitions`.")
