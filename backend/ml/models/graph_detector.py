"""
PCDS Enterprise - Graph Neural Network Detector
Attack chain and lateral movement detection using graph analysis
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict


class GraphConvolutionLayer:
    """Graph Convolution Layer (GCN)"""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Weight matrix
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros(output_dim)
    
    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Forward pass through GCN layer
        
        Args:
            node_features: (num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes) adjacency matrix
        """
        # Normalize adjacency matrix
        D = np.diag(np.sum(adj_matrix, axis=1) + 1e-6)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        adj_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        # Message passing
        aggregated = adj_norm @ node_features
        
        # Linear transformation
        output = aggregated @ self.W + self.b
        
        # ReLU activation
        output = np.maximum(0, output)
        
        return output


class GraphAttentionLayer:
    """Graph Attention Layer (GAT)"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Attention weights
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.a = np.random.randn(2 * output_dim) * 0.1
    
    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with attention"""
        num_nodes = node_features.shape[0]
        
        # Linear transformation
        Wh = node_features @ self.W
        
        # Compute attention scores
        attention_scores = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:  # Only for connected nodes
                    concat = np.concatenate([Wh[i], Wh[j]])
                    score = np.dot(self.a, concat)
                    attention_scores[i, j] = score
        
        # LeakyReLU
        attention_scores = np.where(attention_scores > 0, attention_scores, 0.01 * attention_scores)
        
        # Softmax per node
        attention_weights = np.zeros_like(attention_scores)
        for i in range(num_nodes):
            mask = adj_matrix[i] > 0
            if np.any(mask):
                exp_scores = np.exp(attention_scores[i, mask] - np.max(attention_scores[i, mask]))
                attention_weights[i, mask] = exp_scores / np.sum(exp_scores)
        
        # Aggregate
        output = attention_weights @ Wh
        
        return output, attention_weights


class GlobalAttentionPooling:
    """Global attention pooling for graph-level representation"""
    
    def __init__(self, hidden_dim: int):
        self.gate_nn = np.random.randn(hidden_dim, 1) * 0.1
    
    def forward(self, node_features: np.ndarray) -> np.ndarray:
        """Pool all nodes into single graph representation"""
        # Compute attention scores
        scores = node_features @ self.gate_nn
        attention = np.exp(scores) / np.sum(np.exp(scores))
        
        # Weighted sum
        graph_repr = np.sum(node_features * attention, axis=0)
        
        return graph_repr


class GraphNeuralNetworkDetector:
    """
    Graph Neural Network for attack chain and lateral movement detection
    Analyzes relationships between entities to detect coordinated attacks
    """
    
    def __init__(self, node_features: int = 16, hidden_dim: int = 32, num_layers: int = 3):
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # GCN layers
        self.gcn_layers = [
            GraphConvolutionLayer(node_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ]
        
        # Attention layer
        self.attention_layer = GraphAttentionLayer(hidden_dim, hidden_dim)
        
        # Pooling
        self.pooling = GlobalAttentionPooling(hidden_dim)
        
        # Output
        self.W_out = np.random.randn(hidden_dim, 1) * 0.1
        
        # Threshold
        self.threshold = 0.5
        
        # Entity graph
        self.entity_features = {}
        self.entity_connections = defaultdict(set)
        self.attack_patterns = []
    
    def build_graph(self, entities: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Build graph from entity data"""
        num_nodes = len(entities)
        
        if num_nodes == 0:
            return np.zeros((1, self.node_features)), np.eye(1)
        
        # Build node features
        node_features = np.zeros((num_nodes, self.node_features))
        entity_ids = []
        
        for i, entity in enumerate(entities):
            entity_ids.append(entity.get('id', str(i)))
            node_features[i] = self._extract_node_features(entity)
        
        # Build adjacency matrix based on connections
        adj_matrix = np.eye(num_nodes)  # Self-loops
        
        for i, entity in enumerate(entities):
            connections = entity.get('connections', [])
            for conn in connections:
                if conn in entity_ids:
                    j = entity_ids.index(conn)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Undirected
        
        return node_features, adj_matrix
    
    def _extract_node_features(self, entity: Dict) -> np.ndarray:
        """Extract features for a single node/entity"""
        features = np.zeros(self.node_features)
        
        # Entity type encoding (0-3)
        entity_type = entity.get('type', 'host')
        type_map = {'host': 0, 'user': 1, 'ip': 2, 'process': 3}
        features[0] = type_map.get(entity_type, 0) / 3.0
        
        # Risk level (1-4)
        urgency = entity.get('urgency', 'low')
        urgency_map = {'critical': 1.0, 'high': 0.75, 'medium': 0.5, 'low': 0.25}
        features[1] = urgency_map.get(urgency, 0.25)
        
        # Detection count (normalized)
        features[2] = min(entity.get('detection_count', 0) / 100.0, 1.0)
        
        # Connection count (normalized)
        features[3] = min(entity.get('connection_count', 0) / 50.0, 1.0)
        
        # Is internal
        features[4] = 1.0 if entity.get('is_internal', True) else 0.0
        
        # Has been source of attack
        features[5] = 1.0 if entity.get('attack_source', False) else 0.0
        
        # Has been target of attack
        features[6] = 1.0 if entity.get('attack_target', False) else 0.0
        
        # Time-based features
        features[7] = entity.get('activity_score', 0.5)
        
        # Fill remaining with entity metadata
        for i in range(8, self.node_features):
            features[i] = np.random.random() * 0.1  # Small noise for unused
        
        return features
    
    def forward(self, entities: List[Dict]) -> Tuple[float, Dict]:
        """Forward pass through GNN"""
        node_features, adj_matrix = self.build_graph(entities)
        
        # Pass through GCN layers
        x = node_features
        for gcn in self.gcn_layers:
            x = gcn.forward(x, adj_matrix)
        
        # Attention layer
        x, attention_weights = self.attention_layer.forward(x, adj_matrix)
        
        # Global pooling
        graph_repr = self.pooling.forward(x)
        
        # Output
        logit = np.dot(graph_repr, self.W_out)[0]
        anomaly_score = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
        
        return float(anomaly_score), {
            'attention_weights': attention_weights.tolist(),
            'graph_representation': graph_repr.tolist(),
            'num_nodes': len(entities)
        }
    
    def predict(self, entities: List[Dict]) -> Tuple[bool, float, float]:
        """Predict if graph shows attack pattern"""
        anomaly_score, _ = self.forward(entities)
        is_anomaly = anomaly_score > self.threshold
        confidence = anomaly_score if is_anomaly else (1 - anomaly_score)
        
        return is_anomaly, float(anomaly_score), float(confidence)
    
    def detect_attack_chains(self, entities: List[Dict]) -> List[Dict]:
        """Detect potential attack chains in graph"""
        chains = []
        
        # Build entity lookup
        entity_dict = {e.get('id'): e for e in entities}
        
        # Find high-risk entities
        high_risk = [e for e in entities if e.get('urgency') in ['critical', 'high']]
        
        # Trace connections from high-risk entities
        for entity in high_risk:
            chain = self._trace_attack_chain(entity, entity_dict, set())
            if len(chain) > 1:
                chains.append({
                    'start': entity.get('id'),
                    'path': [e['id'] for e in chain],
                    'length': len(chain),
                    'risk': self._calculate_chain_risk(chain)
                })
        
        return sorted(chains, key=lambda x: x['risk'], reverse=True)[:5]
    
    def _trace_attack_chain(self, entity: Dict, entity_dict: Dict, visited: Set) -> List[Dict]:
        """Recursively trace attack chain"""
        if entity.get('id') in visited:
            return []
        
        visited.add(entity.get('id'))
        chain = [entity]
        
        # Follow connections to other high-risk entities
        for conn_id in entity.get('connections', []):
            if conn_id in entity_dict and conn_id not in visited:
                conn_entity = entity_dict[conn_id]
                if conn_entity.get('urgency') in ['critical', 'high', 'medium']:
                    chain.extend(self._trace_attack_chain(conn_entity, entity_dict, visited))
        
        return chain
    
    def _calculate_chain_risk(self, chain: List[Dict]) -> float:
        """Calculate risk score for attack chain"""
        if not chain:
            return 0.0
        
        urgency_scores = {'critical': 1.0, 'high': 0.75, 'medium': 0.5, 'low': 0.25}
        
        total_risk = sum(
            urgency_scores.get(e.get('urgency', 'low'), 0.25)
            for e in chain
        )
        
        # Longer chains are more concerning
        length_factor = min(len(chain) / 5.0, 1.0)
        
        return (total_risk / len(chain)) * (1 + length_factor)
    
    def detect_lateral_movement(self, entities: List[Dict]) -> Dict:
        """Detect lateral movement patterns"""
        # Count internal-to-internal connections
        internal_hops = 0
        total_hops = 0
        
        entity_dict = {e.get('id'): e for e in entities}
        
        for entity in entities:
            if entity.get('is_internal', True):
                for conn_id in entity.get('connections', []):
                    total_hops += 1
                    if conn_id in entity_dict and entity_dict[conn_id].get('is_internal', True):
                        internal_hops += 1
        
        lateral_ratio = internal_hops / max(total_hops, 1)
        
        return {
            'lateral_movement_detected': lateral_ratio > 0.7,
            'internal_hop_ratio': float(lateral_ratio),
            'total_connections': total_hops,
            'risk_level': 'high' if lateral_ratio > 0.8 else 'medium' if lateral_ratio > 0.6 else 'low'
        }


# Global instance
graph_detector = GraphNeuralNetworkDetector(
    node_features=16,
    hidden_dim=32,
    num_layers=3
)
