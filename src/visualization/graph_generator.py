"""
Graph Generator — creates graph.html, an interactive codebase visualization.
Uses pyvis + networkx to show Python files as nodes with import edges
and pipeline flow as thick red arrows.

Color scheme:
  Blue   (#4A90D9): data pipeline files
  Green  (#5CB85C): training files
  Orange (#F0AD4E): evaluation files
  Purple (#9B59B6): visualization files
  Gray   (#95A5A6): config / utils / root files
"""

import os, ast, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

FILE_COLORS = {
    "data":         "#4A90D9",
    "training":     "#5CB85C",
    "evaluation":   "#F0AD4E",
    "visualization":"#9B59B6",
    "root":         "#95A5A6",
}

PIPELINE_EDGES = [
    ("download_datasets", "prepare_datasets", "Downloads HH-RLHF"),
    ("prepare_datasets", "phase1_sft",        "SFT training data"),
    ("prepare_datasets", "phase2_grpo",       "GRPO prompts"),
    ("phase1_sft",       "phase2_grpo",       "π_ref (SFT model)"),
    ("reward_function",  "phase2_grpo",       "constitutional reward"),
    ("phase2_grpo",      "evaluator",         "GRPO-aligned model"),
    ("phase1_sft",       "evaluator",         "SFT model"),
]


def get_file_category(path: Path) -> str:
    parts = path.parts
    if "data" in parts:         return "data"
    if "training" in parts:     return "training"
    if "evaluation" in parts:   return "evaluation"
    if "visualization" in parts:return "visualization"
    return "root"


def extract_file_info(path: Path) -> dict:
    """Extract docstring and function names from a Python file."""
    try:
        src = path.read_text(errors="ignore")
        tree = ast.parse(src)
        docstring = ast.get_docstring(tree) or ""
        fns = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return {"docstring": docstring[:200], "functions": fns, "imports": imports}
    except Exception:
        return {"docstring": "", "functions": [], "imports": []}


def generate_graph_html(output_path: str = "graph.html"):
    """Generate interactive HTML graph of the codebase."""
    try:
        from pyvis.network import Network
        import networkx as nx
    except ImportError:
        print("[graph] pyvis or networkx not installed. pip install pyvis networkx")
        _generate_fallback_html(output_path)
        return

    net = Network(height="750px", width="100%", bgcolor="#1a1a2e", font_color="#e0e0e0",
                  directed=True, notebook=False)
    net.set_options("""
    {
      "nodes": {"borderWidth": 2, "shadow": true},
      "edges": {"smooth": {"type": "curvedCW", "roundness": 0.2}},
      "physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 200}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    G = nx.DiGraph()
    py_files = list(PROJECT_ROOT.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]

    node_info = {}
    for f in py_files:
        stem = f.stem
        rel = f.relative_to(PROJECT_ROOT)
        cat = get_file_category(f)
        info = extract_file_info(f)
        node_info[stem] = {"path": str(rel), "category": cat, "info": info}

        label = f"{stem}\n({cat})"
        title = (f"<b>{rel}</b><br>"
                 f"<i>{info['docstring'][:100]}{'...' if len(info['docstring'])>100 else ''}</i><br>"
                 f"<b>Functions:</b> {', '.join(info['functions'][:5])}")
        net.add_node(stem, label=stem, title=title,
                     color=FILE_COLORS.get(cat, FILE_COLORS["root"]),
                     size=20 + len(info["functions"]) * 2,
                     shape="box")
        G.add_node(stem)

    # Import-based edges (thin gray)
    all_stems = set(node_info.keys())
    for stem, data in node_info.items():
        for imp in data["info"]["imports"]:
            imp_stem = imp.split(".")[-1]
            if imp_stem in all_stems and imp_stem != stem:
                if not G.has_edge(stem, imp_stem):
                    net.add_edge(stem, imp_stem, color="#555577", width=1, arrows="to")
                    G.add_edge(stem, imp_stem)

    # Pipeline flow edges (thick red)
    for src, dst, label in PIPELINE_EDGES:
        if src in all_stems and dst in all_stems:
            net.add_edge(src, dst, label=label, color="#e74c3c",
                         width=4, arrows="to", font={"color": "#e74c3c"})

    net.save_graph(output_path)
    print(f"[graph] Interactive graph saved → {output_path}")
    print(f"[graph] Nodes: {len(py_files)} files | Pipeline edges: {len(PIPELINE_EDGES)}")


def _generate_fallback_html(output_path: str):
    """Simple fallback HTML if pyvis not installed."""
    py_files = [f for f in PROJECT_ROOT.rglob("*.py") if "__pycache__" not in str(f)]
    items = "".join(f"<li><code>{f.relative_to(PROJECT_ROOT)}</code></li>" for f in py_files)
    html = f"""<!DOCTYPE html><html><head><title>CAI Codebase</title>
    <style>body{{background:#1a1a2e;color:#e0e0e0;font-family:monospace;padding:20px}}
    li{{margin:5px 0}}</style></head>
    <body><h1>Constitutional AI — Codebase Files</h1>
    <p>Install pyvis for interactive graph: <code>pip install pyvis networkx</code></p>
    <ul>{items}</ul></body></html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"[graph] Fallback HTML saved → {output_path}")


if __name__ == "__main__":
    generate_graph_html(str(PROJECT_ROOT / "graph.html"))
