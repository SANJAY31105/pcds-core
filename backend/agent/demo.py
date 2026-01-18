"""
PCDS Agent - Full Local Demo Mode
Captures traffic, extracts features, detects threats locally
No cloud required - perfect for demos and testing
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Optional, List
import json

from capture import PacketCapture, NetworkFlow
from features import FeatureExtractor, FlowFeatures
from analyzer import LocalMLAnalyzer, ThreatDetection

# Color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ASCII Art Banner
BANNER = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     ██████╗  ██████╗██████╗ ███████╗     █████╗  ██████╗ ███████╗  ║
║     ██╔══██╗██╔════╝██╔══██╗██╔════╝    ██╔══██╗██╔════╝ ██╔════╝  ║
║     ██████╔╝██║     ██║  ██║███████╗    ███████║██║  ███╗█████╗    ║
║     ██╔═══╝ ██║     ██║  ██║╚════██║    ██╔══██║██║   ██║██╔══╝    ║
║     ██║     ╚██████╗██████╔╝███████║    ██║  ██║╚██████╔╝███████╗  ║
║     ╚═╝      ╚═════╝╚═════╝ ╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ║
║                                                                    ║
║         Predictive Cyber Defence System - Network Agent           ║
║                    Team SURAKSHA AI © 2026                         ║
╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PCDSLocalAgent:
    """
    Full local agent for demos
    Captures traffic, detects threats, shows results in terminal
    """
    
    def __init__(
        self,
        interface: str = None,
        model_path: str = None,
        verbose: bool = False
    ):
        self.interface = interface
        self.verbose = verbose
        
        # Initialize components
        self.capture = PacketCapture(interface=interface)
        self.extractor = FeatureExtractor()
        self.analyzer = LocalMLAnalyzer(model_path=model_path)
        
        # Statistics
        self.stats = {
            "start_time": None,
            "packets_captured": 0,
            "flows_analyzed": 0,
            "threats_detected": 0,
            "threats_by_type": {}
        }
        
        # Control
        self._running = False
        
        # Threat history (last 100)
        self.threat_history: List[ThreatDetection] = []
    
    def _print_threat(self, threat: ThreatDetection):
        """Print threat alert to terminal"""
        severity_colors = {
            "CRITICAL": Colors.RED + Colors.BOLD,
            "HIGH": Colors.RED,
            "MEDIUM": Colors.WARNING,
            "LOW": Colors.BLUE
        }
        
        color = severity_colors.get(threat.severity, Colors.ENDC)
        
        print(f"\n{color}{'='*70}")
        print(f"  🚨 THREAT DETECTED: {threat.threat_type}")
        print(f"{'='*70}{Colors.ENDC}")
        print(f"  {Colors.BOLD}Severity:{Colors.ENDC}   {color}{threat.severity}{Colors.ENDC}")
        print(f"  {Colors.BOLD}Confidence:{Colors.ENDC} {threat.confidence*100:.1f}%")
        print(f"  {Colors.BOLD}Flow ID:{Colors.ENDC}    {threat.flow_id}")
        print(f"")
        print(f"  {Colors.BOLD}Description:{Colors.ENDC}")
        print(f"    {threat.description}")
        
        if threat.mitre_tactic:
            print(f"")
            print(f"  {Colors.BOLD}MITRE ATT&CK:{Colors.ENDC}")
            print(f"    Tactic: {threat.mitre_tactic}")
            print(f"    Technique: {threat.mitre_technique}")
        
        print(f"")
        print(f"  {Colors.GREEN}{Colors.BOLD}Recommended Action:{Colors.ENDC}")
        print(f"    {threat.recommended_action}")
        print(f"{color}{'='*70}{Colors.ENDC}\n")
    
    def _print_stats(self):
        """Print current statistics"""
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds() if self.stats["start_time"] else 0
        
        print(f"\n{Colors.CYAN}{'─'*70}")
        print(f"  📊 AGENT STATISTICS (Running: {elapsed:.0f}s)")
        print(f"{'─'*70}{Colors.ENDC}")
        print(f"  Packets Captured:  {self.capture.get_stats()['packets_captured']}")
        print(f"  Flows Analyzed:    {self.stats['flows_analyzed']}")
        print(f"  Threats Detected:  {Colors.RED}{self.stats['threats_detected']}{Colors.ENDC}")
        
        if self.stats['threats_by_type']:
            print(f"")
            print(f"  Threats by Type:")
            for ttype, count in self.stats['threats_by_type'].items():
                print(f"    - {ttype}: {count}")
        
        print(f"{Colors.CYAN}{'─'*70}{Colors.ENDC}\n")
    
    def _process_flows(self):
        """Process completed flows and detect threats"""
        flows = self.capture.get_completed_flows()
        
        for flow in flows:
            # Extract features
            features = self.extractor.extract(flow)
            self.stats['flows_analyzed'] += 1
            
            # Analyze for threats
            threat = self.analyzer.analyze(features)
            
            if threat:
                self.stats['threats_detected'] += 1
                self.stats['threats_by_type'][threat.threat_type] = \
                    self.stats['threats_by_type'].get(threat.threat_type, 0) + 1
                
                self.threat_history.append(threat)
                if len(self.threat_history) > 100:
                    self.threat_history.pop(0)
                
                # Print alert
                self._print_threat(threat)
            
            elif self.verbose:
                print(f"  [OK] {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} ({flow.protocol})")
    
    def start(self):
        """Start the local agent"""
        print(BANNER)
        
        print(f"{Colors.GREEN}[*] Starting PCDS Network Agent...{Colors.ENDC}")
        print(f"    Interface: {self.interface or 'Default'}")
        print(f"    ML Model: {self.analyzer.model_type}")
        print(f"")
        
        self._running = True
        self.stats["start_time"] = datetime.now()
        
        # Start capture
        self.capture.start()
        
        print(f"{Colors.GREEN}[*] Agent is running! Monitoring network traffic...{Colors.ENDC}")
        print(f"{Colors.CYAN}[*] Press Ctrl+C to stop and see final statistics.{Colors.ENDC}")
        print(f"")
        
        # Main loop
        stats_interval = 30  # Print stats every 30 seconds
        last_stats_time = time.time()
        
        try:
            while self._running:
                self._process_flows()
                
                # Print periodic stats
                if time.time() - last_stats_time > stats_interval:
                    self._print_stats()
                    last_stats_time = time.time()
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}[!] Shutdown requested...{Colors.ENDC}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the agent"""
        self._running = False
        self.capture.stop()
        
        # Final stats
        print(f"\n{Colors.CYAN}{'═'*70}")
        print(f"  FINAL SESSION REPORT")
        print(f"{'═'*70}{Colors.ENDC}")
        
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds() if self.stats["start_time"] else 0
        capture_stats = self.capture.get_stats()
        
        print(f"  Session Duration:  {elapsed:.0f} seconds")
        print(f"  Packets Captured:  {capture_stats['packets_captured']}")
        print(f"  Packets Dropped:   {capture_stats['packets_dropped']}")
        print(f"  Bytes Captured:    {capture_stats['bytes_captured']:,}")
        print(f"  Flows Created:     {capture_stats['flows_created']}")
        print(f"  Flows Analyzed:    {self.stats['flows_analyzed']}")
        print(f"  {Colors.RED}Threats Detected:  {self.stats['threats_detected']}{Colors.ENDC}")
        
        if self.stats['threats_by_type']:
            print(f"")
            print(f"  Threat Breakdown:")
            for ttype, count in sorted(self.stats['threats_by_type'].items(), key=lambda x: -x[1]):
                print(f"    {Colors.WARNING}{ttype}:{Colors.ENDC} {count}")
        
        print(f"{Colors.CYAN}{'═'*70}{Colors.ENDC}")
        print(f"{Colors.GREEN}[*] Agent stopped. Stay secure! 🛡️{Colors.ENDC}\n")
    
    def export_threats(self, filename: str = "threats.json"):
        """Export detected threats to JSON"""
        with open(filename, 'w') as f:
            json.dump([t.to_dict() for t in self.threat_history], f, indent=2)
        print(f"[*] Threats exported to {filename}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PCDS Network Agent - Local Demo Mode"
    )
    parser.add_argument("-i", "--interface", help="Network interface")
    parser.add_argument("-m", "--model", help="Path to ML model file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all traffic")
    parser.add_argument("-l", "--list", action="store_true", help="List interfaces")
    
    args = parser.parse_args()
    
    if args.list:
        from list_interfaces import list_interfaces
        list_interfaces()
        return
    
    # Create and run agent
    agent = PCDSLocalAgent(
        interface=args.interface,
        model_path=args.model,
        verbose=args.verbose
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        agent.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    agent.start()


if __name__ == "__main__":
    main()
