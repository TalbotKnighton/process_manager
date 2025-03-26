"""Flow graph visualization utilities."""
from __future__ import annotations
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import json
import tempfile
import subprocess
from datetime import datetime
import logging
if TYPE_CHECKING:
    from flow.core.flow import Flow

from flow.core.types import FlowStatus, FlowType

logger = logging.getLogger(__name__)

class FlowVisualizer:
    """Visualizes flow dependencies and execution status."""

    def __init__(self, flow: Flow):
        self.flow = flow
        self.context = flow.context
        self._node_name_map = {}  # Maps process_ids to readable names
        logger.info(f"Initializing FlowVisualizer for flow {flow.config.name}")

    def _get_node_name(self, flow: Flow) -> str:
        """Get a readable node name, creating one if it doesn't exist."""
        if flow.process_id not in self._node_name_map:
            base_name = flow.config.name.lower().replace(" ", "_")
            # If name already exists, append a number
            existing_names = set(self._node_name_map.values())
            if base_name in existing_names:
                counter = 1
                while f"{base_name}_{counter}" in existing_names:
                    counter += 1
                base_name = f"{base_name}_{counter}"
            self._node_name_map[flow.process_id] = base_name
            logger.info(f"Mapped flow ID {flow.process_id} to name {base_name}")
        return self._node_name_map[flow.process_id]

    def to_mermaid(self) -> str:
        """Generate Mermaid graph definition."""
        logger.info("Generating Mermaid graph")
        nodes = []
        edges = []
        
        # Helper to get node style based on status
        def get_node_style(flow: Flow) -> str:
            status_styles = {
                FlowStatus.PENDING: "",
                FlowStatus.RUNNING: "style %s fill:#aff,stroke:#0aa",
                FlowStatus.COMPLETED: "style %s fill:#afa,stroke:#0a0",
                FlowStatus.FAILED: "style %s fill:#faa,stroke:#a00",
                FlowStatus.CANCELLED: "style %s fill:#eee,stroke:#999"
            }
            node_name = self._get_node_name(flow)
            return status_styles[flow.status] % node_name if status_styles[flow.status] else ""

        # Build nodes and edges
        visited = set()
        def visit_flow(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            # Add node with readable name
            node_name = self._get_node_name(flow)
            node_def = f"    {node_name}[\"{flow.config.name}\"]"
            nodes.append(node_def)
            logger.info(f"Added node: {node_def}")
            
            style = get_node_style(flow)
            if style:
                nodes.append(f"    {style}")
            
            # Add edges with readable names
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_name = self._get_node_name(dep_flow)
                    style = " style=dashed" if dep_type == "optional" else ""
                    edge_def = f"    {dep_name} --> {node_name}{style}"
                    edges.append(edge_def)
                    logger.info(f"Added edge: {edge_def}")
                    visit_flow(dep_flow)

        visit_flow(self.flow)
        
        # Build mermaid diagram
        mermaid = ["graph TD;"]
        mermaid.extend(nodes)
        mermaid.extend(edges)
        
        result = "\n".join(mermaid)
        logger.info(f"Generated Mermaid graph:\n{result}")
        return result

    def to_graphviz(self, output_path: Optional[str] = None) -> Optional[str]:
        """Generate Graphviz visualization.
        
        Args:
            output_path: Optional path to save the rendered image.
                        If None, returns the DOT representation.
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        def add_flow_node(flow: Flow):
            status_colors = {
                FlowStatus.PENDING: "white",
                FlowStatus.RUNNING: "lightblue",
                FlowStatus.COMPLETED: "lightgreen",
                FlowStatus.FAILED: "lightcoral",
                FlowStatus.CANCELLED: "lightgray"
            }
            
            G.add_node(
                self._get_node_name(flow),
                label=f"{flow.config.name}\n({flow.status.value})",
                style="filled",
                fillcolor=status_colors[flow.status],
                shape="box",
                fontname="Arial"
            )

        # Add edges with attributes
        def add_flow_edges(flow: Flow):
            flow_name = self._get_node_name(flow)
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_name = self._get_node_name(dep_flow)
                    G.add_edge(
                        dep_name,
                        flow_name,
                        style="dashed" if dep_type == "optional" else "solid",
                        color="gray" if dep_type == "optional" else "black"
                    )

        # Build graph
        visited = set()
        def build_graph(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            add_flow_node(flow)
            add_flow_edges(flow)
            
            for dep_id in flow._dependencies:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    build_graph(dep_flow)

        build_graph(self.flow)
        
        if output_path:
            # Use graphviz to render
            dot_path = Path(output_path).with_suffix('.dot')
            nx.drawing.nx_pydot.write_dot(G, dot_path)
            
            # Convert to desired format
            output_path = Path(output_path)
            format = output_path.suffix.lstrip('.')
            subprocess.run(['dot', '-T' + format, dot_path, '-o', output_path])
            dot_path.unlink()  # Clean up dot file
            return output_path.absolute().as_posix()
        else:
            return nx.drawing.nx_pydot.to_pydot(G).to_string()

    def to_plotly(self, output_path: Optional[str|Path] = None) -> go.Figure:#|Path:
        """Generate interactive Plotly visualization.
        
        Args:
            output_path: Optional path to save as HTML.
                        If None, returns the Figure object.
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        pos = {}  # For node positions
        node_colors = []
        node_labels = {}
        edge_colors = []
        edge_styles = []
        
        status_colors = {
            FlowStatus.PENDING: "#ffffff",
            FlowStatus.RUNNING: "#aaffff",
            FlowStatus.COMPLETED: "#aaffaa",
            FlowStatus.FAILED: "#ffaaaa",
            FlowStatus.CANCELLED: "#eeeeee"
        }
        
        def add_flow_data(flow: Flow):
            node_name = self._get_node_name(flow)
            G.add_node(node_name)
            node_colors.append(status_colors[flow.status])
            node_labels[node_name] = f"{flow.config.name}\n({flow.status.value})"
            
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_name = self._get_node_name(dep_flow)
                    G.add_edge(dep_name, node_name)
                    edge_colors.append("gray" if dep_type == "optional" else "black")
                    edge_styles.append("dash" if dep_type == "optional" else "solid")

        # Build graph data
        visited = set()
        def build_graph_data(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            add_flow_data(flow)
            
            for dep_id in flow._dependencies:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    build_graph_data(dep_flow)

        build_graph_data(self.flow)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create edge traces
        edge_traces = []
        for edge, color, style in zip(G.edges(), edge_colors, edge_styles):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=1, color=color, dash='dash' if style == 'dash' else 'solid'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2)
            ),
            text=[node_labels[node] for node in G.nodes()],
            textposition="bottom center"
        )

        # Create figure
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=f"Flow Graph: {self.flow.config.name}",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        if output_path:
            fig.write_html(output_path)
            # return Path(output_path)
        # else:
        return fig

'''V2
"""Flow graph visualization utilities."""
from __future__ import annotations
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import json
import tempfile
import subprocess
from datetime import datetime
if TYPE_CHECKING:
    from flow.core.flow import Flow

from flow.core.types import FlowStatus, FlowType

class FlowVisualizer:
    """Visualizes flow dependencies and execution status."""

    def __init__(self, flow: Flow):
        self.flow = flow
        self.context = flow.context
        self._node_name_map = {}  # Maps process_ids to readable names

    def _get_node_name(self, flow: Flow) -> str:
        """Get a readable node name, creating one if it doesn't exist."""
        if flow.process_id not in self._node_name_map:
            base_name = flow.config.name.lower().replace(" ", "_")
            # If name already exists, append a number
            existing_names = set(self._node_name_map.values())
            if base_name in existing_names:
                counter = 1
                while f"{base_name}_{counter}" in existing_names:
                    counter += 1
                base_name = f"{base_name}_{counter}"
            self._node_name_map[flow.process_id] = base_name
        return self._node_name_map[flow.process_id]

    def to_mermaid(self) -> str:
        """Generate Mermaid graph definition."""
        nodes = []
        edges = []
        
        # Helper to get node style based on status
        def get_node_style(flow: Flow) -> str:
            status_styles = {
                FlowStatus.PENDING: "",
                FlowStatus.RUNNING: "style %s fill:#aff,stroke:#0aa",
                FlowStatus.COMPLETED: "style %s fill:#afa,stroke:#0a0",
                FlowStatus.FAILED: "style %s fill:#faa,stroke:#a00",
                FlowStatus.CANCELLED: "style %s fill:#eee,stroke:#999"
            }
            node_name = self._get_node_name(flow)
            return status_styles[flow.status] % node_name if status_styles[flow.status] else ""

        # Build nodes and edges
        visited = set()
        def visit_flow(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            # Add node with readable name
            node_name = self._get_node_name(flow)
            nodes.append(f"    {node_name}[{flow.config.name}]")
            style = get_node_style(flow)
            if style:
                nodes.append(f"    {style}")
            
            # Add edges with readable names
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_name = self._get_node_name(dep_flow)
                    style = " style=dashed" if dep_type == "optional" else ""
                    edges.append(f"    {dep_name} --> {node_name}{style}")
                    visit_flow(dep_flow)

        visit_flow(self.flow)
        
        # Build mermaid diagram
        mermaid = ["graph TD;"]
        mermaid.extend(nodes)
        mermaid.extend(edges)
        
        return "\n".join(mermaid)

    def to_graphviz(self, output_path: Optional[str] = None) -> Optional[str]:
        """Generate Graphviz visualization.
        
        Args:
            output_path: Optional path to save the rendered image.
                        If None, returns the DOT representation.
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        def add_flow_node(flow: Flow):
            status_colors = {
                FlowStatus.PENDING: "white",
                FlowStatus.RUNNING: "lightblue",
                FlowStatus.COMPLETED: "lightgreen",
                FlowStatus.FAILED: "lightcoral",
                FlowStatus.CANCELLED: "lightgray"
            }
            
            G.add_node(
                self._get_node_name(flow),
                label=f"{flow.config.name}\n({flow.status.value})",
                style="filled",
                fillcolor=status_colors[flow.status],
                shape="box",
                fontname="Arial"
            )

        # Add edges with attributes
        def add_flow_edges(flow: Flow):
            flow_name = self._get_node_name(flow)
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_name = self._get_node_name(dep_flow)
                    G.add_edge(
                        dep_name,
                        flow_name,
                        style="dashed" if dep_type == "optional" else "solid",
                        color="gray" if dep_type == "optional" else "black"
                    )

        # Build graph
        visited = set()
        def build_graph(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            add_flow_node(flow)
            add_flow_edges(flow)
            
            for dep_id in flow._dependencies:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    build_graph(dep_flow)

        build_graph(self.flow)
        
        if output_path:
            # Use graphviz to render
            dot_path = Path(output_path).with_suffix('.dot')
            nx.drawing.nx_pydot.write_dot(G, dot_path)
            
            # Convert to desired format
            output_path = Path(output_path)
            format = output_path.suffix.lstrip('.')
            subprocess.run(['dot', '-T' + format, dot_path, '-o', output_path])
            dot_path.unlink()  # Clean up dot file
            return output_path.absolute().as_posix()
        else:
            return nx.drawing.nx_pydot.to_pydot(G).to_string()

    def to_plotly(self, output_path: Optional[str] = None) -> Optional[str]:
        """Generate interactive Plotly visualization.
        
        Args:
            output_path: Optional path to save as HTML.
                        If None, returns the Figure object.
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        pos = {}  # For node positions
        node_colors = []
        node_labels = {}
        edge_colors = []
        edge_styles = []
        
        status_colors = {
            FlowStatus.PENDING: "#ffffff",
            FlowStatus.RUNNING: "#aaffff",
            FlowStatus.COMPLETED: "#aaffaa",
            FlowStatus.FAILED: "#ffaaaa",
            FlowStatus.CANCELLED: "#eeeeee"
        }
        
        def add_flow_data(flow: Flow):
            node_name = self._get_node_name(flow)
            G.add_node(node_name)
            node_colors.append(status_colors[flow.status])
            node_labels[node_name] = f"{flow.config.name}\n({flow.status.value})"
            
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    dep_name = self._get_node_name(dep_flow)
                    G.add_edge(dep_name, node_name)
                    edge_colors.append("gray" if dep_type == "optional" else "black")
                    edge_styles.append("dash" if dep_type == "optional" else "solid")

        # Build graph data
        visited = set()
        def build_graph_data(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            add_flow_data(flow)
            
            for dep_id in flow._dependencies:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    build_graph_data(dep_flow)

        build_graph_data(self.flow)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create edge traces
        edge_traces = []
        for edge, color, style in zip(G.edges(), edge_colors, edge_styles):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=1, color=color, dash='dash' if style == 'dash' else 'solid'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2)
            ),
            text=[node_labels[node] for node in G.nodes()],
            textposition="bottom center"
        )

        # Create figure
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=f"Flow Graph: {self.flow.config.name}",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        if output_path:
            fig.write_html(output_path)
            return output_path
        else:
            return fig
'''

'''V1
"""Flow graph visualization utilities."""
from __future__ import annotations
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import json
import tempfile
import subprocess
from datetime import datetime
if TYPE_CHECKING:
    from flow.core.flow import Flow

from flow.core.types import FlowStatus, FlowType

class FlowVisualizer:
    """Visualizes flow dependencies and execution status."""

    def __init__(self, flow: Flow):
        self.flow = flow
        self.context = flow.context

    def to_mermaid(self) -> str:
        """Generate Mermaid graph definition."""
        nodes = []
        edges = []
        
        # Helper to get node style based on status
        def get_node_style(flow: Flow) -> str:
            status_styles = {
                FlowStatus.PENDING: "",
                FlowStatus.RUNNING: "style %s fill:#aff,stroke:#0aa",
                FlowStatus.COMPLETED: "style %s fill:#afa,stroke:#0a0",
                FlowStatus.FAILED: "style %s fill:#faa,stroke:#a00",
                FlowStatus.CANCELLED: "style %s fill:#eee,stroke:#999"
            }
            return status_styles[flow.status] % flow.process_id

        # Build nodes and edges
        visited = set()
        def visit_flow(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            # Add node
            nodes.append(f"    {flow.process_id}[{flow.config.name}]")
            if flow.status != FlowStatus.PENDING:
                nodes.append(f"    {get_node_style(flow)}")
            
            # Add edges
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    style = "style=dashed" if dep_type == "optional" else ""
                    edges.append(f"    {dep_id} --> {flow.process_id} {style}")
                    visit_flow(dep_flow)

        visit_flow(self.flow)
        
        # Build mermaid diagram
        mermaid = ["```mermaid", "graph TD;"]
        mermaid.extend(nodes)
        mermaid.extend(edges)
        mermaid.append("```")
        
        return "\n".join(mermaid)

    def to_graphviz(self, output_path: Optional[str] = None) -> Optional[str]:
        """Generate Graphviz visualization.
        
        Args:
            output_path: Optional path to save the rendered image.
                        If None, returns the DOT representation.
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        def add_flow_node(flow: Flow):
            status_colors = {
                FlowStatus.PENDING: "white",
                FlowStatus.RUNNING: "lightblue",
                FlowStatus.COMPLETED: "lightgreen",
                FlowStatus.FAILED: "lightcoral",
                FlowStatus.CANCELLED: "lightgray"
            }
            
            G.add_node(
                flow.process_id,
                label=f"{flow.config.name}\n({flow.status.value})",
                style="filled",
                fillcolor=status_colors[flow.status],
                shape="box",
                fontname="Arial"
            )

        # Add edges with attributes
        def add_flow_edges(flow: Flow):
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    G.add_edge(
                        dep_id,
                        flow.process_id,
                        style="dashed" if dep_type == "optional" else "solid",
                        color="gray" if dep_type == "optional" else "black"
                    )

        # Build graph
        visited = set()
        def build_graph(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            add_flow_node(flow)
            add_flow_edges(flow)
            
            for dep_id in flow._dependencies:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    build_graph(dep_flow)

        build_graph(self.flow)
        
        if output_path:
            # Use graphviz to render
            dot_path = Path(output_path).with_suffix('.dot')
            nx.drawing.nx_pydot.write_dot(G, dot_path)
            
            # Convert to desired format
            output_path = Path(output_path)
            format = output_path.suffix.lstrip('.')
            subprocess.run(['dot', '-T' + format, dot_path, '-o', output_path])
            dot_path.unlink()  # Clean up dot file
            return output_path.absolute().as_posix()
        else:
            return nx.drawing.nx_pydot.to_pydot(G).to_string()

    def to_plotly(self, output_path: Optional[str] = None) -> Optional[str]:
        """Generate interactive Plotly visualization.
        
        Args:
            output_path: Optional path to save as HTML.
                        If None, returns the Figure object.
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        pos = {}  # For node positions
        node_colors = []
        node_labels = {}
        edge_colors = []
        edge_styles = []
        
        status_colors = {
            FlowStatus.PENDING: "#ffffff",
            FlowStatus.RUNNING: "#aaffff",
            FlowStatus.COMPLETED: "#aaffaa",
            FlowStatus.FAILED: "#ffaaaa",
            FlowStatus.CANCELLED: "#eeeeee"
        }
        
        def add_flow_data(flow: Flow):
            G.add_node(flow.process_id)
            node_colors.append(status_colors[flow.status])
            node_labels[flow.process_id] = f"{flow.config.name}\n({flow.status.value})"
            
            for dep_id, dep_type in flow._dependencies.items():
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    G.add_edge(dep_id, flow.process_id)
                    edge_colors.append("gray" if dep_type == "optional" else "black")
                    edge_styles.append("dash" if dep_type == "optional" else "solid")

        # Build graph data
        visited = set()
        def build_graph_data(flow: Flow):
            if flow.process_id in visited:
                return
            visited.add(flow.process_id)
            
            add_flow_data(flow)
            
            for dep_id in flow._dependencies:
                dep_flow = self.context.get_flow(dep_id)
                if dep_flow:
                    build_graph_data(dep_flow)

        build_graph_data(self.flow)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create edge traces
        edge_traces = []
        for edge, color, style in zip(G.edges(), edge_colors, edge_styles):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=1, color=color, dash='dash' if style == 'dash' else 'solid'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2)
            ),
            text=[node_labels[node] for node in G.nodes()],
            textposition="bottom center"
        )

        # Create figure
        fig = go.Figure(
            data=[*edge_traces, node_trace],
            layout=go.Layout(
                title=f"Flow Graph: {self.flow.config.name}",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        if output_path:
            fig.write_html(output_path)
            return output_path
        else:
            return fig

# Usage Example:
"""
# Create visualizer
visualizer = FlowVisualizer(my_flow)

# Generate Mermaid diagram
mermaid = visualizer.to_mermaid()
print(mermaid)

# Generate and save Graphviz image
visualizer.to_graphviz('flow_graph.png')

# Generate interactive Plotly visualization
visualizer.to_plotly('flow_graph.html')
"""
'''