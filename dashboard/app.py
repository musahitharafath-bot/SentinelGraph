"""
SentinelGraph — Dark Futuristic Security Dashboard
Streamlit frontend for the existing FastAPI + Neo4j backend.

Run:
    python -m streamlit run dashboard/app.py

Optional environment variable:
    API_BASE_URL=http://localhost:8000
"""

import os
import math
import hashlib

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
BG = "#06111F"
PANEL = "#0A1B2E"
PANEL_2 = "#0D2238"
CYAN = "#22D3EE"
BLUE = "#38BDF8"
RED = "#F43F5E"
YELLOW = "#FACC15"
GREEN = "#34D399"
MUTED = "#9FB3C8"
TEXT = "#F1F5F9"
BORDER = "#163B58"


st.set_page_config(
    page_title="SentinelGraph | Cybersecurity Intelligence",
    layout="wide",
    page_icon="🛡️",
)


# -------------------------------------------------------------------
# Theme
# -------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: dark;
        --sg-bg: {BG};
        --sg-panel: {PANEL};
        --sg-cyan: {CYAN};
        --sg-text: {TEXT};
    }}
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background: radial-gradient(ellipse at 50% 0%, #102D49 0%, {BG} 55%);
        color: {TEXT};
    }}
    [data-testid="stHeader"] {{ background: rgba(6,17,31,.88); }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #071525 0%, #030914 100%);
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT}; }}
    h1, h2, h3, h4 {{ color: {TEXT} !important; letter-spacing: -.35px; }}
    p, label, [data-testid="stMarkdownContainer"] {{ color: #D6E4F0; }}
    [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, #102A43, #071626);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 17px 19px;
        box-shadow: 0 0 18px rgba(34,211,238,.07), inset 0 1px 0 rgba(255,255,255,.04);
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {TEXT}; font-weight: 800; }}
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
        background: linear-gradient(135deg, #0E7490, #155E75);
        color: white; border: 1px solid #22D3EE; border-radius: 9px;
        font-weight: 700; transition: .15s ease;
    }}
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{
        background: #164E63; border-color: #67E8F9; color: white;
        box-shadow: 0 0 14px rgba(34,211,238,.22);
    }}
    div[data-baseweb="tab-list"] {{
        gap: 5px; border-bottom: 1px solid {BORDER};
    }}
    button[data-baseweb="tab"] {{
        background: #0A1B2E; border-radius: 8px 8px 0 0;
        color: #B8C9D9; font-weight: 650;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {CYAN}; border-bottom: 3px solid {CYAN};
        background: #102A43;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden;
    }}
    [data-testid="stAlert"] {{ border-radius: 10px; }}
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER}; border-radius: 10px; background: rgba(10,27,46,.7);
    }}
    input, textarea {{
        background-color: #071626 !important; color: {TEXT} !important;
    }}
    .sg-hero {{
        background: linear-gradient(115deg, rgba(13,34,56,.98), rgba(4,15,29,.95));
        border: 1px solid #155E75; border-left: 5px solid {CYAN};
        border-radius: 16px; padding: 25px 28px; margin: 4px 0 22px 0;
        box-shadow: 0 0 25px rgba(34,211,238,.08), inset 0 1px 0 rgba(255,255,255,.05);
    }}
    .sg-eyebrow {{ color: {CYAN}; font-size: 11px; font-weight: 800; letter-spacing: 2.2px; }}
    .sg-title {{ color: #F8FAFC; font-size: 32px; font-weight: 850; margin: 7px 0; }}
    .sg-subtitle {{ color: #B8C9D9; font-size: 14px; margin: 0; }}
    .sg-panel {{
        background: linear-gradient(145deg, rgba(13,34,56,.95), rgba(6,17,31,.96));
        border: 1px solid {BORDER}; border-radius: 13px; padding: 16px 18px;
        box-shadow: 0 0 18px rgba(56,189,248,.035);
    }}
    .sg-label {{ color: {CYAN}; font-size: 11px; font-weight: 800; letter-spacing: 1.5px; }}
    .sg-panel-title {{ color: {TEXT}; font-size: 17px; font-weight: 750; margin: 6px 0; }}
    .sg-muted {{ color: {MUTED}; font-size: 13px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------
def api_get(path: str, params: dict = None):
    try:
        response = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"API error calling {path}: {exc}")
        return None


def api_post(path: str, json_body: dict = None):
    try:
        response = requests.post(
            f"{API_BASE}{path}", json=json_body, timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"API error calling {path}: {exc}")
        return None


def as_list(value):
    return value if isinstance(value, list) else []


def asset_label(asset):
    return str(asset.get("hostname") or asset.get("asset_id") or "Unknown asset")


def risk_color(value):
    value = str(value or "").lower()
    if "critical" in value:
        return RED
    if "high" in value:
        return YELLOW
    if "medium" in value:
        return BLUE
    if "low" in value:
        return GREEN
    return MUTED


def stable_position(identifier, index, count):
    """Deterministic 3D placement; positions are visual only, not real coordinates."""
    digest = hashlib.md5(str(identifier).encode("utf-8")).hexdigest()
    phase = int(digest[:8], 16) / 0xFFFFFFFF * 2 * math.pi
    ring = 1.0 + (int(digest[8:12], 16) % 100) / 100
    z = ((int(digest[12:16], 16) % 200) / 100) - 1
    angle = phase + (2 * math.pi * index / max(count, 1))
    return ring * math.cos(angle), ring * math.sin(angle), z


def path_node_sequence(path):
    """Extract a node sequence only when the API response provides one."""
    if isinstance(path, list):
        return [str(x) for x in path if x is not None]
    if not isinstance(path, dict):
        return []
    for key in ("path_hostnames", "hostnames", "path", "nodes", "asset_ids", "assets"):
        value = path.get(key)
        if isinstance(value, list):
            sequence = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("hostname") or item.get("asset_id") or item.get("id")
                    if name is not None:
                        sequence.append(str(name))
                elif item is not None:
                    sequence.append(str(item))
            if len(sequence) >= 2:
                return sequence
    return []


def make_3d_graph(assets, paths, degree_data):
    """Plot real asset records and only path connections returned by the API."""
    graph = nx.Graph()
    for asset in assets:
        node_id = asset.get("asset_id")
        if node_id is not None:
            graph.add_node(str(node_id), **asset)

    id_by_name = {}
    for node_id, data in graph.nodes(data=True):
        id_by_name[node_id] = node_id
        hostname = data.get("hostname")
        if hostname is not None:
            id_by_name[str(hostname)] = node_id

    for path in paths:
        sequence = path_node_sequence(path)
        mapped = [id_by_name[node] for node in sequence if node in id_by_name]
        for source, target in zip(mapped, mapped[1:]):
            if source != target:
                graph.add_edge(source, target, source="API attack path")

    # Degree information is useful for sizing, but is not treated as edge data.
    degree_map = {}
    for row in degree_data:
        if isinstance(row, dict) and row.get("asset_id") is not None:
            degree_map[str(row["asset_id"])] = row.get("degree", 0)

    return graph, degree_map


def plot_3d_graph(graph, degree_map):
    nodes = list(graph.nodes())
    if not nodes:
        st.info("No asset records are available to plot.")
        return

    positions = {
        node: stable_position(node, i, len(nodes))
        for i, node in enumerate(nodes)
    }

    edge_x, edge_y, edge_z = [], [], []
    for source, target in graph.edges():
        x1, y1, z1 = positions[source]
        x2, y2, z2 = positions[target]
        edge_x.extend([x1, x2, None])
        edge_y.extend([y1, y2, None])
        edge_z.extend([z1, z2, None])

    fig = go.Figure()
    if graph.number_of_edges():
        fig.add_trace(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(color="rgba(34,211,238,.55)", width=3),
                hoverinfo="skip",
                name="API-returned attack-path links",
            )
        )

    node_x, node_y, node_z, labels, hovers, colors, sizes = [], [], [], [], [], [], []
    for node in nodes:
        x, y, z = positions[node]
        data = graph.nodes[node]
        criticality = str(data.get("criticality", "unknown")).lower()
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        labels.append(str(data.get("hostname") or node))
        hovers.append(
            f"Asset: {data.get('hostname', node)}<br>"
            f"ID: {node}<br>"
            f"Criticality: {data.get('criticality', 'Not provided')}<br>"
            f"Degree: {degree_map.get(node, 'Not provided')}"
        )
        colors.append(risk_color(criticality))
        try:
            sizes.append(8 + min(14, int(degree_map.get(node, 0) or 0) * 2))
        except (ValueError, TypeError):
            sizes.append(9)

    fig.add_trace(
        go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.95,
                line=dict(color="#DDFBFF", width=1),
            ),
            name="Assets",
        )
    )
    fig.update_layout(
        height=590,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Arial"),
        showlegend=False,
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.05)),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if graph.number_of_edges() == 0:
        st.caption(
            "The API supplied asset records but no usable node sequences in "
            "the returned attack paths. Nodes are shown in a 3D layout; no "
            "connections have been invented. To draw real links, the backend "
            "must expose graph edges or path node sequences."
        )
    else:
        st.caption(
            f"Showing {graph.number_of_nodes()} API assets and "
            f"{graph.number_of_edges()} links extracted from API-returned paths. "
            "Node positions are a visual layout, not geographic coordinates."
        )


# -------------------------------------------------------------------
# App header and admin sidebar
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="sg-hero">
        <div class="sg-eyebrow">CYBERSECURITY INTELLIGENCE PLATFORM</div>
        <div class="sg-title">SENTINELGRAPH</div>
        <p class="sg-subtitle">
            Attack-path discovery · Graph analytics · Risk intelligence · DNN predictions
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ SentinelGraph")
    st.caption("SECURITY OPERATIONS CONSOLE")
    st.divider()
    st.subheader("Admin controls")

    if st.button("Initialize schema", use_container_width=True):
        result = api_post("/admin/init-schema")
        if result:
            st.success(result.get("status", "Schema initialization completed"))
            st.json(result)

    if st.button("Run full ingestion", use_container_width=True):
        with st.spinner("Ingesting source data into Neo4j..."):
            result = api_post("/admin/ingest")
        if result:
            st.success("Ingestion request completed")
            st.json(result)

    if st.button("Train DNN risk model", use_container_width=True):
        with st.spinner("Training model..."):
            result = api_post("/ml/train")
        if result:
            st.success("Training request completed")
            st.json(result.get("metrics", result))

    st.divider()
    st.markdown("**Backend connection**")
    st.caption(API_BASE)
    if st.button("Check API", use_container_width=True):
        health = api_get("/openapi.json")
        if health:
            st.success("FastAPI is responding")
        else:
            st.warning("Could not contact FastAPI")


tabs = st.tabs([
    "Overview",
    "Network Graph",
    "Attack Paths",
    "Risk Scores",
    "Centrality",
    "DNN Predictions",
    "Ask the Graph (LLM)",
])


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="sg-label">LIVE SECURITY POSTURE</div>', unsafe_allow_html=True)
    st.subheader("Executive overview")

    surface = api_get("/analysis/attack-surface")
    assets_response = api_get("/assets")
    surface_ok = isinstance(surface, dict)
    assets_ok = isinstance(assets_response, list)
    surface = surface if surface_ok else {}
    assets = assets_response if assets_ok else []
    exposed = as_list(surface.get("exposed_critical_assets"))
    paths = as_list(surface.get("internet_to_critical_paths"))
    route_count = surface.get("num_attack_routes", "—") if surface_ok else "—"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Internet → Critical Routes", route_count)
    with m2:
        st.metric("Exposed Critical / High Assets", len(exposed) if surface_ok else "—")
    with m3:
        st.metric("Total Assets", len(assets) if assets_ok else "—")
    with m4:
        st.metric("Returned Attack Paths", len(paths) if surface_ok else "—")

    if not surface_ok and not assets_ok:
        st.warning("Overview data is unavailable. Check FastAPI and the Neo4j connection.")
    elif not assets and not exposed and not paths:
        st.info("The API returned no assets, exposed assets, or paths. If this is unexpected, run ingestion and verify the source data.")

    st.markdown("")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="sg-panel"><div class="sg-label">GRAPH INTELLIGENCE</div><div class="sg-panel-title">Interactive 3D attack topology</div><div class="sg-muted">Drag to rotate · Scroll to zoom · Hover for asset details</div></div>', unsafe_allow_html=True)
        degree_data = as_list(api_get("/analysis/centrality/degree"))
        graph, degree_map = make_3d_graph(assets, paths, degree_data)
        plot_3d_graph(graph, degree_map)

    with right:
        st.markdown('<div class="sg-panel"><div class="sg-label">PRIORITY WATCHLIST</div><div class="sg-panel-title">Exposed critical / high assets</div></div>', unsafe_allow_html=True)
        if exposed:
            st.dataframe(pd.DataFrame(exposed), use_container_width=True, hide_index=True)
        elif surface_ok:
            st.info("No exposed critical/high assets were returned by the API.")
        else:
            st.caption("Exposure data unavailable.")

        st.markdown('<div class="sg-panel"><div class="sg-label">ATTACK SURFACE</div><div class="sg-panel-title">Internet-to-critical paths</div></div>', unsafe_allow_html=True)
        if paths:
            st.dataframe(pd.DataFrame(paths), use_container_width=True, hide_index=True)
        elif surface_ok:
            st.info("No attack paths were returned by the API.")
        else:
            st.caption("Path data unavailable.")

    st.divider()
    st.markdown('<div class="sg-label">ASSET INTELLIGENCE</div>', unsafe_allow_html=True)
    st.subheader("Asset inventory")
    if assets:
        asset_df = pd.DataFrame(assets)
        if "criticality" in asset_df.columns:
            priority = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            asset_df["_priority"] = asset_df["criticality"].astype(str).str.lower().map(priority).fillna(4)
            asset_df = asset_df.sort_values("_priority").drop(columns=["_priority"])
        st.dataframe(asset_df, use_container_width=True, hide_index=True)
    elif assets_ok:
        st.info("No assets returned. Run ingestion if source data is ready.")


# -------------------------------------------------------------------
# Network Graph
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Interactive 3D Network Topology")
    assets = as_list(api_get("/assets"))
    paths_data = as_list((api_get("/analysis/attack-surface") or {}).get("internet_to_critical_paths"))
    degree_data = as_list(api_get("/analysis/centrality/degree"))
    graph, degree_map = make_3d_graph(assets, paths_data, degree_data)
    if graph.number_of_nodes():
        plot_3d_graph(graph, degree_map)
        st.markdown("**Legend:** red = critical · yellow = high · cyan = medium/unknown · green = low")
    else:
        st.info("No asset records available. Run ingestion and check the API.")


# -------------------------------------------------------------------
# Attack Paths
# -------------------------------------------------------------------
with tabs[2]:
    st.subheader("Attack Path Explorer")
    assets = as_list(api_get("/assets"))
    ids = [
        f"{a.get('asset_id', '')} — {a.get('hostname', '')}"
        for a in assets if a.get("asset_id") is not None
    ]
    id_map = {f"{a.get('asset_id', '')} — {a.get('hostname', '')}": a["asset_id"] for a in assets if a.get("asset_id") is not None}

    if not ids:
        st.info("No assets available for path exploration.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            start_sel = st.selectbox("Start asset", ids, key="start_asset")
        with c2:
            end_sel = st.selectbox("End asset", ids, index=min(3, len(ids) - 1), key="end_asset")
        with c3:
            max_hops = st.slider("Maximum hops", 1, 10, 6)

        if st.button("Find shortest path"):
            result = api_post("/analysis/shortest-path", {"start_id": id_map[start_sel], "end_id": id_map[end_sel]})
            if result:
                if result.get("found", True) and result.get("path_hostnames"):
                    st.success(" → ".join(result.get("path_hostnames", [])))
                else:
                    st.warning(result.get("message", "No path found between selected assets."))
                st.json(result)

        if st.button("Find all paths ranked by risk"):
            result = api_post("/analysis/all-paths", {"start_id": id_map[start_sel], "end_id": id_map[end_sel], "max_hops": max_hops})
            if result:
                st.dataframe(pd.DataFrame(result if isinstance(result, list) else [result]), use_container_width=True)


# -------------------------------------------------------------------
# Risk Scores
# -------------------------------------------------------------------
with tabs[3]:
    st.subheader("Composite Risk Scores")
    scores = api_get("/analysis/risk-scores")
    if scores:
        df = pd.DataFrame(scores)
        st.dataframe(df.drop(columns=["components"], errors="ignore"), use_container_width=True, hide_index=True)
        if {"asset_id", "risk_score"}.issubset(df.columns):
            fig = px.bar(
                df.head(15), x="asset_id", y="risk_score",
                color="criticality" if "criticality" in df.columns else None,
                title="Top assets by composite risk score",
                color_discrete_map={"critical": RED, "high": YELLOW, "medium": BLUE, "low": GREEN},
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk score data returned. Check ingestion and API status.")

    st.subheader("Top Critical Assets (risk + centrality blend)")
    top_n = st.slider("Top N assets", 3, 20, 10)
    critical = api_get("/analysis/critical-assets", params={"top_n": top_n})
    if critical:
        st.dataframe(pd.DataFrame(critical if isinstance(critical, list) else [critical]), use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# Centrality
# -------------------------------------------------------------------
with tabs[4]:
    st.subheader("Graph Centrality")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PageRank — structural importance**")
        pr = api_get("/analysis/centrality/pagerank")
        if pr:
            st.dataframe(pd.DataFrame(pr), use_container_width=True, hide_index=True)
        else:
            st.info("No PageRank results returned.")
    with c2:
        st.markdown("**Betweenness — bridge / choke-point assets**")
        bw = api_get("/analysis/centrality/betweenness")
        if bw:
            st.dataframe(pd.DataFrame(bw), use_container_width=True, hide_index=True)
        else:
            st.info("No betweenness results returned.")


# -------------------------------------------------------------------
# DNN Predictions
# -------------------------------------------------------------------
with tabs[5]:
    st.subheader("DNN Compromise Risk Predictions")
    st.caption("Predictions returned by the trained model on the API server.")
    preds = api_get("/ml/predict")
    if preds:
        df = pd.DataFrame(preds if isinstance(preds, list) else [preds])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if {"asset_id", "predicted_compromise_probability"}.issubset(df.columns):
            fig = px.bar(
                df.head(15), x="asset_id", y="predicted_compromise_probability",
                color="predicted_label" if "predicted_label" in df.columns else None,
                title="Predicted compromise probability by asset",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No prediction data returned. Train the DNN risk model from the sidebar first.")


# -------------------------------------------------------------------
# Ask the Graph (LLM)
# -------------------------------------------------------------------
with tabs[6]:
    st.subheader("Ask the Attack Graph")
    st.caption("Natural language → Cypher → Neo4j → answer. Requires GROQ_API_KEY on the API server.")

    question = st.text_input(
        "Ask a question about the attack graph",
        placeholder="Which internet-exposed assets have critical vulnerabilities with an exploit available?",
    )
    if st.button("Ask the graph") and question.strip():
        with st.spinner("Querying the graph..."):
            result = api_post("/llm/ask", {"question": question.strip()})
        if result:
            st.markdown("**Answer**")
            st.write(result.get("answer", "No answer field returned."))
            with st.expander("Generated Cypher and raw results"):
                st.code(result.get("cypher_query", ""), language="cypher")
                st.json(result.get("results", []))

    st.divider()
    st.subheader("Executive risk summary")
    if st.button("Generate executive summary"):
        with st.spinner("Preparing summary..."):
            summary = api_get("/llm/executive-summary")
        if summary:
            st.markdown(summary.get("narrative", "No narrative returned."))
            with st.expander("Underlying asset data"):
                st.dataframe(pd.DataFrame(summary.get("assets", [])), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Explain a specific attack path")
    assets = as_list(api_get("/assets"))
    valid_assets = [a for a in assets if a.get("asset_id") is not None]
    labels = [f"{a.get('asset_id', '')} — {a.get('hostname', '')}" for a in valid_assets]
    id_map = {f"{a.get('asset_id', '')} — {a.get('hostname', '')}": a["asset_id"] for a in valid_assets}

    if not labels:
        st.info("No assets are available to select.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_sel = st.selectbox("Start asset", labels, key="llm_start")
        with c2:
            end_sel = st.selectbox("End asset", labels, key="llm_end")
        if st.button("Explain this path"):
            with st.spinner("Generating path explanation..."):
                result = api_post("/llm/explain-path", {"start_id": id_map[start_sel], "end_id": id_map[end_sel]})
            if result:
                st.markdown(result.get("narrative", "No narrative returned."))
