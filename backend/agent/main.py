"""
PCDS Network Agent - Main Entry Point
Captures network traffic from SPAN port and sends to PCDS Cloud
"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional
import os

try:
    from .capture import PacketCapture
    from .features import FeatureExtractor
    from .sender import CloudSender, AgentConfig, LocalCache
except ImportError:
    from capture import PacketCapture
    from features import FeatureExtractor
    from sender import CloudSender, AgentConfig, LocalCache


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pcds_agent.log')
    ]
)
logger = logging.getLogger(__name__)


class PCDSAgent:
    """
    Main PCDS Network Agent
    Orchestrates capture, feature extraction, and cloud sending
    """
    
    def __init__(
        self,
        interface: str,
        api_key: str,
        api_endpoint: str = "https://api.pcds.app/v1",
        agent_id: str = None,
        organization_id: str = ""
    ):
        # Generate agent ID if not provided
        if not agent_id:
            import uuid
            agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        
        self.agent_id = agent_id
        
        # Initialize components
        self.capture = PacketCapture(
            interface=interface,
            bpf_filter="ip",  # Capture all IP traffic
            packet_callback=self._on_packet
        )
        
        self.feature_extractor = FeatureExtractor()
        
        self.sender = CloudSender(AgentConfig(
            agent_id=agent_id,
            api_key=api_key,
            api_endpoint=api_endpoint,
            organization_id=organization_id
        ))
        
        self.cache = LocalCache()
        
        # State
        self._running = False
        self._flow_count = 0
        
        logger.info(f"PCDS Agent initialized: {agent_id}")
        logger.info(f"Interface: {interface}")
        logger.info(f"API Endpoint: {api_endpoint}")
    
    def _on_packet(self, parsed_packet):
        """Callback for each parsed packet"""
        # Packets are handled by flow aggregation in capture module
        pass
    
    def _process_flows(self):
        """Process completed flows"""
        flows = self.capture.get_completed_flows()
        
        if flows:
            # Extract features
            features = self.feature_extractor.extract_batch(flows)
            
            # Queue for sending
            self.sender.queue_batch(features)
            
            self._flow_count += len(flows)
            
            if len(flows) > 0:
                logger.debug(f"Processed {len(flows)} flows (total: {self._flow_count})")
    
    def start(self):
        """Start the agent"""
        logger.info("=" * 50)
        logger.info("  PCDS Network Agent Starting")
        logger.info("=" * 50)
        
        self._running = True
        
        # Start components
        self.capture.start()
        self.sender.start()
        
        logger.info("Agent is running. Press Ctrl+C to stop.")
        
        # Main loop
        try:
            while self._running:
                self._process_flows()
                time.sleep(0.5)  # Process flows every 500ms
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the agent"""
        logger.info("Stopping agent...")
        
        self._running = False
        
        # Stop components
        self.capture.stop()
        self.sender.stop()
        
        # Print final stats
        self._print_stats()
        
        logger.info("Agent stopped.")
    
    def _print_stats(self):
        """Print agent statistics"""
        capture_stats = self.capture.get_stats()
        sender_stats = self.sender.get_stats()
        
        logger.info("=" * 50)
        logger.info("  Session Statistics")
        logger.info("=" * 50)
        logger.info(f"Packets captured: {capture_stats['packets_captured']}")
        logger.info(f"Packets dropped:  {capture_stats['packets_dropped']}")
        logger.info(f"Bytes captured:   {capture_stats['bytes_captured']}")
        logger.info(f"Flows created:    {capture_stats['flows_created']}")
        logger.info(f"Flows processed:  {self._flow_count}")
        logger.info(f"Batches sent:     {sender_stats['batches_sent']}")
        logger.info(f"Batches failed:   {sender_stats['batches_failed']}")
        logger.info("=" * 50)
    
    def get_status(self) -> dict:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "running": self._running,
            "flows_processed": self._flow_count,
            "capture": self.capture.get_stats(),
            "sender": self.sender.get_stats()
        }


def list_interfaces():
    """List available network interfaces"""
    try:
        from scapy.all import get_if_list, get_if_hwaddr
        print("\nAvailable Network Interfaces:")
        print("-" * 40)
        for iface in get_if_list():
            try:
                mac = get_if_hwaddr(iface)
                print(f"  {iface} ({mac})")
            except:
                print(f"  {iface}")
        print("-" * 40)
    except ImportError:
        print("Scapy not installed. Run: pip install scapy")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PCDS Network Agent - Enterprise Network Traffic Monitor"
    )
    
    parser.add_argument(
        "-i", "--interface",
        help="Network interface to capture from (e.g., eth0, Ethernet)"
    )
    parser.add_argument(
        "-k", "--api-key",
        help="PCDS API key for authentication"
    )
    parser.add_argument(
        "-e", "--endpoint",
        default="https://api.pcds.app/v1",
        help="PCDS API endpoint URL"
    )
    parser.add_argument(
        "-a", "--agent-id",
        help="Unique agent identifier"
    )
    parser.add_argument(
        "-o", "--org-id",
        default="",
        help="Organization ID"
    )
    parser.add_argument(
        "-l", "--list-interfaces",
        action="store_true",
        help="List available network interfaces"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list_interfaces:
        list_interfaces()
        return
    
    if not args.interface:
        print("Error: Network interface required. Use -i/--interface")
        print("Use -l to list available interfaces.")
        sys.exit(1)
    
    if not args.api_key:
        # Check environment variable
        args.api_key = os.environ.get("PCDS_API_KEY", "")
        if not args.api_key:
            print("Warning: No API key provided. Agent will cache data locally.")
            args.api_key = "demo-key"
    
    # Create and run agent
    agent = PCDSAgent(
        interface=args.interface,
        api_key=args.api_key,
        api_endpoint=args.endpoint,
        agent_id=args.agent_id,
        organization_id=args.org_id
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        agent.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start agent
    agent.start()


if __name__ == "__main__":
    main()
