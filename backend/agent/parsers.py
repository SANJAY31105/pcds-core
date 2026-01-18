"""
PCDS Protocol Parsers
Deep packet inspection for application-layer protocols
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class HTTPRequest:
    """Parsed HTTP request"""
    method: str
    path: str
    host: str
    version: str = "1.1"
    headers: Dict[str, str] = field(default_factory=dict)
    user_agent: Optional[str] = None
    content_length: int = 0
    suspicious_patterns: List[str] = field(default_factory=list)


@dataclass
class DNSQuery:
    """Parsed DNS query"""
    query_type: str  # A, AAAA, MX, TXT, etc.
    domain: str
    subdomain_count: int = 0
    is_suspicious: bool = False
    reason: Optional[str] = None


@dataclass
class TLSHandshake:
    """Parsed TLS handshake info"""
    version: str
    cipher_suite: Optional[str] = None
    sni_hostname: Optional[str] = None
    is_suspicious: bool = False


class ProtocolParser:
    """
    Deep packet inspection for common protocols
    Extracts metadata and detects suspicious patterns
    """
    
    # Suspicious HTTP patterns (injection, traversal, etc.)
    HTTP_SUSPICIOUS_PATTERNS = [
        (r'\.\./', "Path traversal"),
        (r'<script', "XSS attempt"),
        (r'UNION.*SELECT', "SQL injection"),
        (r"'.*OR.*'", "SQL injection"),
        (r'/etc/passwd', "File disclosure"),
        (r'/proc/self', "Proc access"),
        (r'cmd\.exe|powershell', "Command injection"),
        (r'eval\(|exec\(', "Code injection"),
        (r'\$\{.*\}', "Template injection"),
    ]
    
    # Suspicious DNS patterns
    DNS_SUSPICIOUS_PATTERNS = [
        (r'^[a-f0-9]{32,}\.', "Possible DNS tunnel"),
        (r'\.onion$', "Tor hidden service"),
        (r'\.i2p$', "I2P network"),
        (r'dyndns|no-ip|afraid\.org', "Dynamic DNS"),
    ]
    
    # Known malicious TLDs
    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work']
    
    def __init__(self):
        logger.info("ProtocolParser initialized")
    
    def parse_http(self, payload: bytes) -> Optional[HTTPRequest]:
        """Parse HTTP request from raw payload"""
        try:
            # Decode payload
            text = payload.decode('utf-8', errors='ignore')
            lines = text.split('\r\n')
            
            if not lines:
                return None
            
            # Parse request line
            request_line = lines[0].split(' ')
            if len(request_line) < 3:
                return None
            
            method, path, version = request_line[0], request_line[1], request_line[2]
            
            # Parse headers
            headers = {}
            host = ""
            user_agent = None
            content_length = 0
            
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    headers[key] = value
                    
                    if key == 'host':
                        host = value
                    elif key == 'user-agent':
                        user_agent = value
                    elif key == 'content-length':
                        try:
                            content_length = int(value)
                        except ValueError:
                            pass
            
            # Check for suspicious patterns
            suspicious = []
            full_request = f"{method} {path} {text}"
            
            for pattern, reason in self.HTTP_SUSPICIOUS_PATTERNS:
                if re.search(pattern, full_request, re.IGNORECASE):
                    suspicious.append(reason)
            
            return HTTPRequest(
                method=method,
                path=path,
                host=host,
                version=version.replace('HTTP/', ''),
                headers=headers,
                user_agent=user_agent,
                content_length=content_length,
                suspicious_patterns=suspicious
            )
            
        except Exception as e:
            logger.debug(f"HTTP parse error: {e}")
            return None
    
    def parse_dns(self, payload: bytes) -> Optional[DNSQuery]:
        """Parse DNS query from raw payload"""
        try:
            # DNS header is 12 bytes
            if len(payload) < 12:
                return None
            
            # Skip header, parse question section
            offset = 12
            domain_parts = []
            
            while offset < len(payload):
                length = payload[offset]
                if length == 0:
                    break
                offset += 1
                part = payload[offset:offset+length].decode('ascii', errors='ignore')
                domain_parts.append(part)
                offset += length
            
            if not domain_parts:
                return None
            
            domain = '.'.join(domain_parts)
            subdomain_count = len(domain_parts) - 2 if len(domain_parts) > 2 else 0
            
            # Check for suspicious patterns
            is_suspicious = False
            reason = None
            
            for pattern, sus_reason in self.DNS_SUSPICIOUS_PATTERNS:
                if re.search(pattern, domain, re.IGNORECASE):
                    is_suspicious = True
                    reason = sus_reason
                    break
            
            # Check TLD
            for tld in self.SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    is_suspicious = True
                    reason = f"Suspicious TLD: {tld}"
                    break
            
            # Check subdomain length (possible DNS tunneling)
            if subdomain_count > 5 or len(domain_parts[0]) > 32:
                is_suspicious = True
                reason = "Excessive subdomains/length (DNS tunnel?)"
            
            return DNSQuery(
                query_type="A",  # Simplified
                domain=domain,
                subdomain_count=subdomain_count,
                is_suspicious=is_suspicious,
                reason=reason
            )
            
        except Exception as e:
            logger.debug(f"DNS parse error: {e}")
            return None
    
    def parse_tls(self, payload: bytes) -> Optional[TLSHandshake]:
        """Parse TLS ClientHello for SNI"""
        try:
            # TLS record header
            if len(payload) < 5:
                return None
            
            content_type = payload[0]
            if content_type != 0x16:  # Handshake
                return None
            
            version_major = payload[1]
            version_minor = payload[2]
            version = f"{version_major}.{version_minor}"
            
            # Try to extract SNI (simplified)
            sni = None
            
            # Look for SNI extension pattern
            try:
                text = payload.decode('latin-1')
                # Simple pattern match for domain-like strings
                matches = re.findall(r'[a-z0-9][a-z0-9\-\.]{5,}[a-z0-9]\.[a-z]{2,}', text, re.IGNORECASE)
                if matches:
                    sni = matches[0]
            except:
                pass
            
            return TLSHandshake(
                version=version,
                sni_hostname=sni
            )
            
        except Exception as e:
            logger.debug(f"TLS parse error: {e}")
            return None
    
    def analyze_payload(self, payload: bytes, dst_port: int) -> Dict[str, Any]:
        """Analyze payload based on port/protocol"""
        result = {
            "protocol": "unknown",
            "suspicious": False,
            "details": {}
        }
        
        if dst_port == 80 or dst_port == 8080:
            http = self.parse_http(payload)
            if http:
                result["protocol"] = "HTTP"
                result["details"] = {
                    "method": http.method,
                    "path": http.path,
                    "host": http.host,
                    "user_agent": http.user_agent
                }
                if http.suspicious_patterns:
                    result["suspicious"] = True
                    result["threats"] = http.suspicious_patterns
        
        elif dst_port == 53:
            dns = self.parse_dns(payload)
            if dns:
                result["protocol"] = "DNS"
                result["details"] = {
                    "domain": dns.domain,
                    "subdomain_count": dns.subdomain_count
                }
                if dns.is_suspicious:
                    result["suspicious"] = True
                    result["reason"] = dns.reason
        
        elif dst_port == 443:
            tls = self.parse_tls(payload)
            if tls:
                result["protocol"] = "TLS"
                result["details"] = {
                    "version": tls.version,
                    "sni": tls.sni_hostname
                }
        
        return result
