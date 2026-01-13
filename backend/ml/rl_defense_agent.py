"""
Reinforcement Learning for Adaptive Cyber Defense
Based on Paper 4: RL agent learns optimal defense strategies

Features:
- Q-Learning based defense policy
- Adaptive response to evolving threats
- Reward-based learning from incident outcomes
- Dynamic action selection (block, isolate, alert, etc.)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random
import json


class DefenseAction(Enum):
    """Available defense actions"""
    MONITOR = 0      # Continue monitoring
    ALERT = 1        # Send alert
    BLOCK_IP = 2     # Block source IP
    ISOLATE = 3      # Isolate affected host
    QUARANTINE = 4   # Quarantine file/process
    FULL_LOCKDOWN = 5  # Emergency lockdown


class ThreatLevel(Enum):
    """Observed threat levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RLState:
    """Environment state for RL agent"""
    threat_level: ThreatLevel
    attack_type: int  # Encoded attack type
    affected_hosts: int  # Number of affected hosts
    time_since_detection: int  # Minutes since first detection
    previous_action: DefenseAction
    action_success_rate: float  # Recent action success rate


@dataclass
class RLExperience:
    """Experience tuple for replay"""
    state: RLState
    action: DefenseAction
    reward: float
    next_state: RLState
    done: bool


class QLearningDefenseAgent:
    """
    Q-Learning based defense agent
    Learns optimal response policies through interaction with environment
    """
    
    def __init__(self, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Exploration rate
        
        # State space dimensions
        self.threat_levels = len(ThreatLevel)
        self.attack_types = 16  # Number of attack classes
        self.host_buckets = 5  # 0, 1-5, 6-20, 21-100, 100+
        self.time_buckets = 4  # 0-5min, 5-30min, 30min-2hr, 2hr+
        
        # Action space
        self.n_actions = len(DefenseAction)
        
        # Initialize Q-table
        # State: (threat_level, attack_type, host_bucket, time_bucket)
        self.q_table = np.zeros((
            self.threat_levels,
            self.attack_types,
            self.host_buckets,
            self.time_buckets,
            self.n_actions
        ))
        
        # Experience replay buffer
        self.experience_buffer: List[RLExperience] = []
        self.max_buffer_size = 10000
        
        # Training stats
        self.episodes = 0
        self.total_reward = 0
        self.actions_taken = {a.name: 0 for a in DefenseAction}
        
        print("🤖 RL Defense Agent initialized")
    
    def _state_to_indices(self, state: RLState) -> Tuple[int, int, int, int]:
        """Convert state to Q-table indices"""
        threat_idx = state.threat_level.value
        attack_idx = min(state.attack_type, self.attack_types - 1)
        
        # Bucket hosts
        if state.affected_hosts == 0:
            host_idx = 0
        elif state.affected_hosts <= 5:
            host_idx = 1
        elif state.affected_hosts <= 20:
            host_idx = 2
        elif state.affected_hosts <= 100:
            host_idx = 3
        else:
            host_idx = 4
        
        # Bucket time
        if state.time_since_detection <= 5:
            time_idx = 0
        elif state.time_since_detection <= 30:
            time_idx = 1
        elif state.time_since_detection <= 120:
            time_idx = 2
        else:
            time_idx = 3
        
        return threat_idx, attack_idx, host_idx, time_idx
    
    def select_action(self, state: RLState, training: bool = True) -> DefenseAction:
        """
        Select action using epsilon-greedy policy
        """
        indices = self._state_to_indices(state)
        
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # Select best action from Q-table
            action_idx = np.argmax(self.q_table[indices])
        
        action = DefenseAction(action_idx)
        self.actions_taken[action.name] += 1
        
        return action
    
    def update(self, state: RLState, action: DefenseAction, 
               reward: float, next_state: RLState, done: bool):
        """
        Update Q-table using Q-learning update rule
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        """
        state_indices = self._state_to_indices(state)
        next_indices = self._state_to_indices(next_state)
        
        current_q = self.q_table[state_indices][action.value]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_indices])
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning update
        self.q_table[state_indices][action.value] += self.lr * (target_q - current_q)
        
        # Store experience
        experience = RLExperience(state, action, reward, next_state, done)
        self._add_experience(experience)
        
        self.total_reward += reward
    
    def _add_experience(self, experience: RLExperience):
        """Add experience to replay buffer"""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def replay_update(self, batch_size: int = 32):
        """Learn from past experiences"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        for exp in batch:
            self.update(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
    
    def calculate_reward(self, action: DefenseAction, outcome: Dict) -> float:
        """
        Calculate reward based on action and outcome
        
        Positive rewards for:
        - Successful threat mitigation
        - Minimal false positives
        - Fast response
        
        Negative rewards for:
        - Failed mitigation
        - Business disruption
        - Unnecessary lockdowns
        """
        reward = 0.0
        
        # Base reward for threat mitigation
        if outcome.get("threat_mitigated", False):
            reward += 10.0
        elif outcome.get("threat_escalated", False):
            reward -= 15.0
        
        # Penalty for false positives
        if outcome.get("false_positive", False):
            reward -= 5.0
            # Extra penalty for aggressive false actions
            if action in [DefenseAction.ISOLATE, DefenseAction.FULL_LOCKDOWN]:
                reward -= 10.0
        
        # Reward for proportionate response
        threat_level = outcome.get("threat_level", ThreatLevel.LOW)
        action_severity = {
            DefenseAction.MONITOR: 0,
            DefenseAction.ALERT: 1,
            DefenseAction.BLOCK_IP: 2,
            DefenseAction.ISOLATE: 3,
            DefenseAction.QUARANTINE: 3,
            DefenseAction.FULL_LOCKDOWN: 4
        }
        
        severity_diff = abs(threat_level.value - action_severity[action])
        if severity_diff == 0:
            reward += 5.0  # Perfect match
        elif severity_diff == 1:
            reward += 2.0  # Close match
        else:
            reward -= severity_diff * 2.0  # Disproportionate
        
        # Speed bonus
        response_time = outcome.get("response_time_seconds", 60)
        if response_time < 10:
            reward += 3.0
        elif response_time < 60:
            reward += 1.0
        
        return reward
    
    def get_policy_recommendation(self, state: RLState) -> Dict:
        """Get recommended action with Q-values"""
        indices = self._state_to_indices(state)
        q_values = self.q_table[indices]
        
        best_action_idx = np.argmax(q_values)
        best_action = DefenseAction(best_action_idx)
        
        # Get all action Q-values
        action_values = {
            a.name: float(q_values[a.value]) 
            for a in DefenseAction
        }
        
        return {
            "recommended_action": best_action.name,
            "confidence": float(np.max(q_values) / (np.sum(np.abs(q_values)) + 1e-6)),
            "q_values": action_values,
            "state": {
                "threat_level": state.threat_level.name,
                "attack_type": state.attack_type,
                "affected_hosts": state.affected_hosts,
                "time_since_detection": state.time_since_detection
            }
        }
    
    def train_episode(self, initial_state: RLState, 
                     max_steps: int = 10) -> Dict:
        """
        Train for one episode (simulated)
        """
        state = initial_state
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = self.select_action(state, training=True)
            
            # Simulate outcome (in production, this would come from actual results)
            outcome = self._simulate_outcome(state, action)
            reward = self.calculate_reward(action, outcome)
            
            # Get next state
            next_state = self._get_next_state(state, action, outcome)
            done = outcome.get("threat_mitigated", False) or step == max_steps - 1
            
            self.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        self.episodes += 1
        
        return {
            "episode": self.episodes,
            "steps": steps,
            "total_reward": episode_reward,
            "final_state": state.threat_level.name
        }
    
    def _simulate_outcome(self, state: RLState, action: DefenseAction) -> Dict:
        """Simulate outcome of action (for training)"""
        # Higher threat level + stronger action = better mitigation
        mitigation_prob = min(0.9, 0.3 + 0.15 * action.value)
        
        if state.threat_level == ThreatLevel.CRITICAL and action == DefenseAction.MONITOR:
            mitigation_prob = 0.1
        
        threat_mitigated = random.random() < mitigation_prob
        
        # False positive probability (lower for higher threats)
        fp_prob = max(0.05, 0.3 - 0.05 * state.threat_level.value)
        false_positive = random.random() < fp_prob and state.threat_level.value <= 1
        
        return {
            "threat_mitigated": threat_mitigated,
            "threat_escalated": not threat_mitigated and random.random() < 0.3,
            "false_positive": false_positive,
            "threat_level": state.threat_level,
            "response_time_seconds": random.randint(5, 120)
        }
    
    def _get_next_state(self, state: RLState, action: DefenseAction, 
                       outcome: Dict) -> RLState:
        """Get next state after action"""
        if outcome.get("threat_mitigated"):
            new_threat = ThreatLevel.NONE
            new_hosts = 0
        elif outcome.get("threat_escalated"):
            new_threat = ThreatLevel(min(4, state.threat_level.value + 1))
            new_hosts = min(100, state.affected_hosts + random.randint(1, 5))
        else:
            new_threat = state.threat_level
            new_hosts = state.affected_hosts
        
        return RLState(
            threat_level=new_threat,
            attack_type=state.attack_type,
            affected_hosts=new_hosts,
            time_since_detection=state.time_since_detection + 5,
            previous_action=action,
            action_success_rate=0.8 if outcome.get("threat_mitigated") else 0.5
        )
    
    def save_policy(self, path: str):
        """Save Q-table to file"""
        np.save(path, self.q_table)
    
    def load_policy(self, path: str):
        """Load Q-table from file"""
        self.q_table = np.load(path)
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "episodes_trained": self.episodes,
            "total_reward": float(self.total_reward),
            "average_reward": float(self.total_reward / max(self.episodes, 1)),
            "experience_buffer_size": len(self.experience_buffer),
            "actions_taken": self.actions_taken,
            "exploration_rate": self.epsilon,
            "learning_rate": self.lr
        }


# Global instance
_agent: Optional[QLearningDefenseAgent] = None


def get_rl_agent() -> QLearningDefenseAgent:
    """Get or create RL defense agent"""
    global _agent
    if _agent is None:
        _agent = QLearningDefenseAgent()
    return _agent
