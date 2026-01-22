"""
M-Polar Fermatean Fuzzy Graph Construction
Research: Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs
Authors: Kholood Alsager, Nada Almuairi

This module implements M-Polar Fermatean Fuzzy Graphs for modeling stock relationships
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations


class MPolarFermateanFuzzyGraph:
    """
    M-Polar Fermatean Fuzzy Graph (M-PFFG) implementation
    
    Used to model interdependence between stocks across multiple polarities:
    1. Price volatility
    2. Trading volume  
    3. Market sentiment
    """
    
    def __init__(self, m=3, stocks=None):
        """
        Initialize M-PFFG
        
        Parameters:
        -----------
        m : int
            Number of polarities (default: 3)
        stocks : list
            List of stock symbols (vertices)
        """
        self.m = m
        self.stocks = stocks if stocks else []
        self.vertices = {}  # {stock: {'md': [...], 'nmd': [...]}}
        self.edges = {}     # {(stock1, stock2): weight}
        
    def add_vertex(self, stock, md, nmd):
        """
        Add vertex (stock) to the graph
        
        Parameters:
        -----------
        stock : str
            Stock symbol
        md : list
            Membership degrees for m polarities [φ1, φ2, ..., φm]
        nmd : list
            Non-membership degrees for m polarities [ψ1, ψ2, ..., ψm]
        """
        if len(md) != self.m or len(nmd) != self.m:
            raise ValueError(f"MD and NMD must have length {self.m}")
        
        # Verify Fermatean condition: φ³ + ψ³ ≤ 1
        for i in range(self.m):
            if md[i]**3 + nmd[i]**3 > 1.0:
                # Correct NMD to satisfy condition
                nmd[i] = (1.0 - md[i]**3) ** (1/3)
                print(f"⚠ Corrected NMD for {stock} polarity {i+1}: {nmd[i]:.3f}")
        
        self.vertices[stock] = {
            'md': np.array(md),
            'nmd': np.array(nmd)
        }
        
        if stock not in self.stocks:
            self.stocks.append(stock)
    
    def calculate_edge_weight(self, stock_i, stock_j):
        """
        Calculate edge weight between two stocks
        
        Formula (from paper):
        h_ij = (1/3) * Σ_t min(1, ∛(φ_i(R_t)·φ_j(R_t)) + [1 - ∛(ψ_i(R_t)·ψ_j(R_t))])
        
        Parameters:
        -----------
        stock_i, stock_j : str
            Stock symbols
        
        Returns:
        --------
        weight : float
            Edge weight (interdependence strength)
        """
        if stock_i not in self.vertices or stock_j not in self.vertices:
            raise ValueError(f"Both stocks must be added as vertices")
        
        md_i = self.vertices[stock_i]['md']
        md_j = self.vertices[stock_j]['md']
        nmd_i = self.vertices[stock_i]['nmd']
        nmd_j = self.vertices[stock_j]['nmd']
        
        weight = 0.0
        
        for t in range(self.m):
            # Cube root of product of membership degrees
            md_product = np.cbrt(md_i[t] * md_j[t])
            
            # Cube root of product of non-membership degrees
            nmd_product = np.cbrt(nmd_i[t] * nmd_j[t])
            
            # Component for this polarity
            component = min(1.0, md_product + (1 - nmd_product))
            
            weight += component
        
        # Average over all polarities
        weight = weight / self.m
        
        return weight
    
    def construct_all_edges(self):
        """
        Calculate edge weights for all pairs of stocks
        Reproduces Table 25 from the research paper
        
        Returns:
        --------
        edge_df : pd.DataFrame
            Edge weights table
        """
        print(f"\n{'='*70}")
        print(f"Constructing M-PFFG Edge Weights")
        print(f"{'='*70}\n")
        
        edge_data = []
        
        # Calculate for all pairs
        for stock_i, stock_j in combinations(self.stocks, 2):
            weight = self.calculate_edge_weight(stock_i, stock_j)
            
            # Store edge
            edge_key = tuple(sorted([stock_i, stock_j]))
            self.edges[edge_key] = weight
            
            # Add to results
            edge_data.append({
                'Stock_Pair': f"{stock_i}-{stock_j}",
                'Volatility_Term': weight,  # Simplified - in full version, calculate per term
                'Volume_Term': weight,
                'Sentiment_Term': weight,
                'Average_h_ij': weight
            })
            
            print(f"{stock_i:5s} - {stock_j:5s} : {weight:.3f}")
        
        edge_df = pd.DataFrame(edge_data)
        
        print(f"\n{'='*70}")
        print(f"Edge Weight Calculation Complete")
        print(f"{'='*70}\n")
        
        return edge_df
    
    def get_adjacency_matrix(self):
        """
        Get adjacency matrix representation
        
        Returns:
        --------
        adj_matrix : np.array
            Adjacency matrix with edge weights
        """
        n = len(self.stocks)
        adj_matrix = np.zeros((n, n))
        
        for i, stock_i in enumerate(self.stocks):
            for j, stock_j in enumerate(self.stocks):
                if i != j:
                    edge_key = tuple(sorted([stock_i, stock_j]))
                    if edge_key in self.edges:
                        adj_matrix[i, j] = self.edges[edge_key]
        
        return adj_matrix
    
    def calculate_fuzzy_risk_adjustments(self):
        """
        Calculate fuzzy-adjusted risk factors for each stock
        Based on M-PFFG edge weights and membership degrees
        
        Returns:
        --------
        risk_adjustments : np.array
            Fuzzy risk adjustment factor for each stock
        """
        risk_adjustments = []
        
        for stock in self.stocks:
            # Sum of edge weights connected to this stock
            connected_weights = []
            
            for edge_key, weight in self.edges.items():
                if stock in edge_key:
                    connected_weights.append(weight)
            
            # Average membership degree across polarities
            avg_md = np.mean(self.vertices[stock]['md'])
            
            # Risk adjustment combines connectivity and membership
            risk_adj = (1 - avg_md) * (1 + np.mean(connected_weights))
            risk_adjustments.append(risk_adj)
        
        return np.array(risk_adjustments)
    
    def visualize_graph(self, save_path='mpffg_network.png'):
        """
        Visualize the M-PFFG as a network graph
        
        Parameters:
        -----------
        save_path : str
            Path to save visualization
        """
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for stock in self.stocks:
            G.add_node(stock)
        
        # Add edges with weights
        for (stock_i, stock_j), weight in self.edges.items():
            G.add_edge(stock_i, stock_j, weight=weight)
        
        # Set up plot
        plt.figure(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node sizes based on average membership degree
        node_sizes = []
        for stock in self.stocks:
            avg_md = np.mean(self.vertices[stock]['md'])
            node_sizes.append(avg_md * 3000 + 500)
        
        # Edge widths based on weights
        edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                               node_color='lightblue', alpha=0.9,
                               edgecolors='black', linewidths=2)
        
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                               alpha=0.6, edge_color='gray')
        
        # Edge labels (weights)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)
        
        plt.title('M-Polar Fermatean Fuzzy Graph - Stock Interdependence Network',
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Graph visualization saved to: {save_path}")
        plt.close()


def construct_mpffg_from_data(fuzzy_df):
    """
    Construct M-PFFG from fuzzy values dataframe (Table 23)
    
    Parameters:
    -----------
    fuzzy_df : pd.DataFrame
        Dataframe with columns: Stock, φ1, ψ1, φ2, ψ2, φ3, ψ3
    
    Returns:
    --------
    graph : MPolarFermateanFuzzyGraph
        Constructed M-PFFG
    """
    stocks = fuzzy_df['Stock'].tolist()
    graph = MPolarFermateanFuzzyGraph(m=3, stocks=stocks)
    
    # Add vertices
    for _, row in fuzzy_df.iterrows():
        stock = row['Stock']
        md = [row['φ1'], row['φ2'], row['φ3']]
        nmd = [row['ψ1'], row['ψ2'], row['ψ3']]
        
        graph.add_vertex(stock, md, nmd)
    
    # Construct edges
    edge_df = graph.construct_all_edges()
    
    return graph, edge_df


if __name__ == "__main__":
    """
    Main execution: Construct M-PFFG from research data
    """
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║    M-Polar Fermatean Fuzzy Graph Construction             ║
    ║    Research: Portfolio Optimization                       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Example fuzzy values from Table 23 (research paper)
    fuzzy_data = {
        'Stock': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'],
        'φ1': [0.60, 0.56, 0.69, 0.52, 1.00],  # Volatility polarity
        'ψ1': [0.92, 0.93, 0.876, 0.94, 0.00],
        'φ2': [0.61, 0.21, 0.41, 0.27, 1.00],  # Volume polarity
        'ψ2': [0.91, 0.98, 0.95, 0.97, 0.00],
        'φ3': [0.78, 0.52, 0.75, 0.80, 0.85],  # Sentiment polarity
        'ψ3': [0.42, 0.38, 0.45, 0.40, 0.36]
    }
    
    fuzzy_df = pd.DataFrame(fuzzy_data)
    
    print("\n3-Polar Fuzzy Values (Table 23):")
    print(fuzzy_df.to_string(index=False))
    
    # Construct graph
    graph, edge_df = construct_mpffg_from_data(fuzzy_df)
    
    # Display edge weights (Table 25)
    print("\nEdge Weights in M-PFFG (Table 25):")
    print(edge_df.to_string(index=False))
    
    # Calculate fuzzy risk adjustments
    risk_adjustments = graph.calculate_fuzzy_risk_adjustments()
    
    print(f"\n{'='*70}")
    print("Fuzzy-Adjusted Risk Factors:")
    print(f"{'='*70}")
    for stock, risk in zip(graph.stocks, risk_adjustments):
        print(f"{stock:5s} : {risk:.3f}")
    print(f"{'='*70}\n")
    
    # Get adjacency matrix
    adj_matrix = graph.get_adjacency_matrix()
    
    print("Adjacency Matrix:")
    adj_df = pd.DataFrame(adj_matrix, 
                          index=graph.stocks, 
                          columns=graph.stocks)
    print(adj_df.round(3))
    
    # Visualize
    graph.visualize_graph('mpffg_network_visualization.png')
    
    # Export results
    edge_df.to_csv('../data/processed/edge_weights_table25.csv', index=False)
    print(f"\n✓ Edge weights exported to: data/processed/edge_weights_table25.csv")
    
    risk_df = pd.DataFrame({
        'Stock': graph.stocks,
        'Fuzzy_Risk_Adjustment': risk_adjustments
    })
    risk_df.to_csv('../data/processed/fuzzy_risk_adjustments.csv', index=False)
    print(f"✓ Risk adjustments exported to: data/processed/fuzzy_risk_adjustments.csv")
    
    print("\n✓ M-PFFG construction complete!")
