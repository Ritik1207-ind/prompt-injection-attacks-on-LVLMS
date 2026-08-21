"""
generators/t2_structural.py
────────────────────────────
Type 2 — Structural Layout Injection
Embeds the malicious instruction inside a visually complex structure
(mind map, flowchart, annotated diagram). The harmful content is
present but camouflaged within surrounding legitimate nodes/labels.
Black-box, zero model access. Stealthiness: LOW.
"""

import random
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
from PIL import Image


# Legitimate-looking node labels for camouflage
CAMOUFLAGE_NODES = {
    "mindmap": [
        "Project Overview", "Key Objectives", "Timeline", "Resources",
        "Budget Planning", "Risk Assessment", "Stakeholders", "Deliverables",
        "Milestones", "Team Structure", "Dependencies", "Success Metrics",
        "Phase 1: Research", "Phase 2: Development", "Review & Iterate",
        "Documentation", "Testing Strategy", "Deployment Plan",
    ],
    "flowchart": [
        "Start", "Receive Input", "Validate Data", "Process Request",
        "Check Conditions", "Generate Output", "Log Results", "End",
        "Error Handling", "Retry Logic", "Cache Check", "API Call",
        "Database Query", "Format Response", "Send Notification",
    ],
    "annotated_diagram": [
        "Component A", "Component B", "Data Layer", "API Gateway",
        "Load Balancer", "Cache Layer", "Auth Service", "Monitoring",
        "Storage", "Message Queue", "Worker Pool", "Frontend",
    ],
}


class StructuralGenerator:
    def __init__(self, cfg: dict, images_dir: Path):
        t2 = cfg["attack_types"]["t2_structural"]
        self.styles = t2.get("styles", ["mindmap", "flowchart", "annotated_diagram"])
        self.camouflage_count = t2.get("camouflage_node_count", 8)
        self.out_dir = images_dir / "t2_structural"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _build_mindmap_dot(self, instruction: str, style: str) -> str:
        """Build a Graphviz DOT string with instruction as one camouflaged node."""
        nodes = random.sample(CAMOUFLAGE_NODES[style], min(self.camouflage_count, len(CAMOUFLAGE_NODES[style])))
        malicious_label = instruction[:80] + ("..." if len(instruction) > 80 else "")

        # Insert malicious node at a random position among legitimate ones
        insert_pos = random.randint(0, len(nodes))
        nodes.insert(insert_pos, malicious_label)

        dot_lines = [
            'digraph G {',
            '  graph [fontname="Helvetica", bgcolor="white", pad="0.5"];',
            '  node [shape=box, style="rounded,filled", fillcolor="lightblue", '
            'fontsize=10, fontname="Helvetica", width=2];',
            '  edge [color="#555555"];',
            '  root [label="Main Topic", shape=ellipse, fillcolor="#FFD700", fontsize=12];',
        ]

        for i, label in enumerate(nodes):
            safe_label = label.replace('"', '\\"').replace('\n', ' ')
            color = "#FFB3B3" if label == malicious_label else "lightblue"
            dot_lines.append(
                f'  n{i} [label="{safe_label}", fillcolor="{color}"];'
            )

        # Connect all nodes to root
        for i in range(len(nodes)):
            dot_lines.append(f'  root -> n{i};')

        # Add some cross-connections for legitimacy
        if len(nodes) > 3:
            for _ in range(random.randint(1, 3)):
                a, b = random.sample(range(len(nodes)), 2)
                dot_lines.append(f'  n{a} -> n{b} [style=dashed, color="#AAAAAA"];')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def _build_flowchart_dot(self, instruction: str) -> str:
        """Build a flowchart with malicious instruction as a disguised step."""
        nodes = random.sample(CAMOUFLAGE_NODES["flowchart"], min(self.camouflage_count, 8))
        malicious_label = instruction[:60] + ("..." if len(instruction) > 60 else "")

        insert_pos = random.randint(1, len(nodes) - 1)
        nodes.insert(insert_pos, malicious_label)

        dot_lines = [
            'digraph G {',
            '  graph [fontname="Helvetica", bgcolor="white", rankdir=TB];',
            '  node [shape=box, style="rounded,filled", fillcolor="#E8F4FD", fontsize=9, fontname="Helvetica"];',
            '  edge [color="#333333", fontsize=8];',
        ]

        for i, label in enumerate(nodes):
            safe_label = label.replace('"', '\\"')
            shape = "ellipse" if i == 0 or i == len(nodes) - 1 else "box"
            color = "#FFB3B3" if label == malicious_label else "#E8F4FD"
            dot_lines.append(f'  n{i} [label="{safe_label}", shape={shape}, fillcolor="{color}"];')

        for i in range(len(nodes) - 1):
            dot_lines.append(f'  n{i} -> n{i+1};')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def generate(self, instruction: str, sample_id: str) -> Path:
        """
        Generate a structural diagram with injection camouflaged inside.
        Returns path to saved image.
        """
        style = random.choice(self.styles)

        if style == "flowchart":
            dot_src = self._build_flowchart_dot(instruction)
        else:
            dot_src = self._build_mindmap_dot(instruction, "mindmap")

        out_path = self.out_dir / f"{sample_id}.png"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as tmp:
            tmp.write(dot_src)
            tmp_path = tmp.name

        try:
            subprocess.run(
                ["dot", "-Tpng", "-o", str(out_path), tmp_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz failed: {e.stderr.decode()}") from e
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Resize to standard size for consistency
        img = Image.open(out_path)
        img = img.resize((800, 600), Image.LANCZOS)
        img.save(out_path, "PNG")

        return out_path
