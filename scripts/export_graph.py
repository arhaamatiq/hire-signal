"""
Export the compiled LangGraph pipeline as a PNG and Mermaid diagram.

Usage:
    uv run scripts/export_graph.py

Outputs:
    assets/pipeline_graph.png   — visual diagram (if graphviz installed)
    stdout                      — raw Mermaid markdown (always)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.agentic_workflow import GraphBuilder


def main() -> int:
    builder = GraphBuilder(model_provider="groq")
    graph = builder()

    mermaid_text = graph.get_graph().draw_mermaid()
    print(mermaid_text)

    assets_dir = REPO_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    png_path = assets_dir / "pipeline_graph.png"

    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"\nWrote PNG: {png_path}", file=sys.stderr)
    except Exception as e:
        print(
            f"\n[export_graph] Could not render PNG ({e.__class__.__name__}: {e}).\n"
            "Tip: paste the Mermaid markdown above into https://mermaid.live to view/export the diagram.",
            file=sys.stderr,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
