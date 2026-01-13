"""
Threat Intelligence API - EDR Enhancement
Integrates with external threat intel sources

Supports:
- VirusTotal API (file/URL/IP reputation)
- AbuseIPDB (IP reputation)
- Local IOC database
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not installed. Threat intel API limited.")


@dataclass
class ThreatIntelResult:
    """Threat intelligence lookup result"""
    indicator: str
    indicator_type: str  # ip, domain, hash, url
    is_malicious: bool
    confidence: float  # 0.0 - 1.0
    sources: List[str]
    details: Dict = field(default_factory=dict)
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class ThreatIntelAPI:
    """
    Threat Intelligence API Integration
    
    Provides:
    - VirusTotal lookups (files, URLs, IPs, domains)
    - AbuseIPDB lookups (IP reputation)
    - Local IOC database
    - Caching for performance
    """
    
    def __init__(self, 
                 virustotal_key: str = None,
                 abuseipdb_key: str = None,
                 cache_ttl: int = 3600):
        """
        Initialize Threat Intel API
        
        Args:
            virustotal_key: VirusTotal API key
            abuseipdb_key: AbuseIPDB API key
            cache_ttl: Cache time-to-live in seconds
        """
        # API Keys (from environment or params)
        self.virustotal_key = virustotal_key or os.environ.get("VIRUSTOTAL_API_KEY", "")
        self.abuseipdb_key = abuseipdb_key or os.environ.get("ABUSEIPDB_API_KEY", "")
        
        # Cache
        self._cache: Dict[str, ThreatIntelResult] = {}
        self.cache_ttl = cache_ttl
        
        # Local IOC database
        self.ioc_db = self._load_local_iocs()
        
        # Stats
        self.stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "vt_lookups": 0,
            "abuseipdb_lookups": 0,
            "local_hits": 0,
            "malicious_found": 0
        }
        
        print("🔍 Threat Intel API initialized")
        if self.virustotal_key:
            print("   ✅ VirusTotal API configured")
        if self.abuseipdb_key:
            print("   ✅ AbuseIPDB API configured")
        print(f"   📦 Local IOC DB: {len(self.ioc_db)} indicators")
    
    def _load_local_iocs(self) -> Dict[str, Dict]:
        """Load local IOC database"""
        iocs = {}
        
        # Known malicious IPs (sample)
        malicious_ips = [
            "185.220.101.1", "185.220.101.2",  # Tor exit nodes
            "23.129.64.0",  # Cobalt Strike C2
            "45.33.32.156",  # Metasploit default
        ]
        for ip in malicious_ips:
            iocs[ip] = {"type": "ip", "threat": "Known malicious", "confidence": 0.9}
        
        # Known malicious domains
        malicious_domains = [
            "malware.com", "evil.com", "c2server.xyz",
        ]
        for domain in malicious_domains:
            iocs[domain] = {"type": "domain", "threat": "Known malicious", "confidence": 0.9}
        
        # Known malware hashes (sample)
        malware_hashes = [
            "44d88612fea8a8f36de82e1278abb02f",  # EICAR test file
        ]
        for h in malware_hashes:
            iocs[h.lower()] = {"type": "hash", "threat": "Known malware", "confidence": 1.0}
        
        return iocs
    
    def check_ip(self, ip: str) -> ThreatIntelResult:
        """Check IP reputation"""
        self.stats["total_lookups"] += 1
        
        # Check cache
        cached = self._get_cached(f"ip:{ip}")
        if cached:
            return cached
        
        sources = []
        is_malicious = False
        confidence = 0.0
        details = {}
        
        # Check local IOC database
        if ip in self.ioc_db:
            self.stats["local_hits"] += 1
            ioc = self.ioc_db[ip]
            sources.append("local_ioc")
            is_malicious = True
            confidence = ioc["confidence"]
            details["local"] = ioc
        
        # Check AbuseIPDB
        if self.abuseipdb_key and REQUESTS_AVAILABLE:
            try:
                abuseipdb_result = self._check_abuseipdb(ip)
                if abuseipdb_result:
                    sources.append("abuseipdb")
                    abuse_score = abuseipdb_result.get("abuseConfidenceScore", 0)
                    details["abuseipdb"] = abuseipdb_result
                    
                    if abuse_score > 50:
                        is_malicious = True
                        confidence = max(confidence, abuse_score / 100)
            except:
                pass
        
        # Check VirusTotal
        if self.virustotal_key and REQUESTS_AVAILABLE:
            try:
                vt_result = self._check_virustotal_ip(ip)
                if vt_result:
                    sources.append("virustotal")
                    details["virustotal"] = vt_result
                    
                    malicious = vt_result.get("malicious", 0)
                    total = vt_result.get("total", 1)
                    
                    if malicious > 0:
                        is_malicious = True
                        confidence = max(confidence, malicious / total)
            except:
                pass
        
        if is_malicious:
            self.stats["malicious_found"] += 1
        
        result = ThreatIntelResult(
            indicator=ip,
            indicator_type="ip",
            is_malicious=is_malicious,
            confidence=confidence,
            sources=sources,
            details=details
        )
        
        self._set_cache(f"ip:{ip}", result)
        return result
    
    def check_domain(self, domain: str) -> ThreatIntelResult:
        """Check domain reputation"""
        self.stats["total_lookups"] += 1
        
        # Check cache
        cached = self._get_cached(f"domain:{domain}")
        if cached:
            return cached
        
        sources = []
        is_malicious = False
        confidence = 0.0
        details = {}
        
        # Check local IOC database
        if domain.lower() in self.ioc_db:
            self.stats["local_hits"] += 1
            ioc = self.ioc_db[domain.lower()]
            sources.append("local_ioc")
            is_malicious = True
            confidence = ioc["confidence"]
            details["local"] = ioc
        
        # Check VirusTotal
        if self.virustotal_key and REQUESTS_AVAILABLE:
            try:
                vt_result = self._check_virustotal_domain(domain)
                if vt_result:
                    sources.append("virustotal")
                    details["virustotal"] = vt_result
                    
                    malicious = vt_result.get("malicious", 0)
                    if malicious > 0:
                        is_malicious = True
                        confidence = max(confidence, min(malicious / 10, 1.0))
            except:
                pass
        
        if is_malicious:
            self.stats["malicious_found"] += 1
        
        result = ThreatIntelResult(
            indicator=domain,
            indicator_type="domain",
            is_malicious=is_malicious,
            confidence=confidence,
            sources=sources,
            details=details
        )
        
        self._set_cache(f"domain:{domain}", result)
        return result
    
    def check_hash(self, file_hash: str) -> ThreatIntelResult:
        """Check file hash reputation"""
        self.stats["total_lookups"] += 1
        file_hash = file_hash.lower()
        
        # Check cache
        cached = self._get_cached(f"hash:{file_hash}")
        if cached:
            return cached
        
        sources = []
        is_malicious = False
        confidence = 0.0
        details = {}
        
        # Check local IOC database
        if file_hash in self.ioc_db:
            self.stats["local_hits"] += 1
            ioc = self.ioc_db[file_hash]
            sources.append("local_ioc")
            is_malicious = True
            confidence = ioc["confidence"]
            details["local"] = ioc
        
        # Check VirusTotal
        if self.virustotal_key and REQUESTS_AVAILABLE:
            try:
                vt_result = self._check_virustotal_hash(file_hash)
                if vt_result:
                    sources.append("virustotal")
                    details["virustotal"] = vt_result
                    
                    malicious = vt_result.get("malicious", 0)
                    total = vt_result.get("total", 1)
                    
                    if malicious > 2:  # At least 3 detections
                        is_malicious = True
                        confidence = max(confidence, malicious / total)
            except:
                pass
        
        if is_malicious:
            self.stats["malicious_found"] += 1
        
        result = ThreatIntelResult(
            indicator=file_hash,
            indicator_type="hash",
            is_malicious=is_malicious,
            confidence=confidence,
            sources=sources,
            details=details
        )
        
        self._set_cache(f"hash:{file_hash}", result)
        return result
    
    def check_file(self, file_path: str) -> ThreatIntelResult:
        """Check file by calculating its hash"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return self.check_hash(file_hash)
        except Exception as e:
            return ThreatIntelResult(
                indicator=file_path,
                indicator_type="file",
                is_malicious=False,
                confidence=0.0,
                sources=[],
                details={"error": str(e)}
            )
    
    def _check_abuseipdb(self, ip: str) -> Optional[Dict]:
        """Query AbuseIPDB API"""
        if not self.abuseipdb_key:
            return None
        
        self.stats["abuseipdb_lookups"] += 1
        
        url = f"https://api.abuseipdb.com/api/v2/check"
        headers = {
            "Key": self.abuseipdb_key,
            "Accept": "application/json"
        }
        params = {"ipAddress": ip, "maxAgeInDays": 90}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("data", {})
        
        return None
    
    def _check_virustotal_ip(self, ip: str) -> Optional[Dict]:
        """Query VirusTotal IP lookup"""
        if not self.virustotal_key:
            return None
        
        self.stats["vt_lookups"] += 1
        
        url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
        headers = {"x-apikey": self.virustotal_key}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get("data", {}).get("attributes", {})
            stats = data.get("last_analysis_stats", {})
            return {
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "total": sum(stats.values()) if stats else 0
            }
        
        return None
    
    def _check_virustotal_domain(self, domain: str) -> Optional[Dict]:
        """Query VirusTotal domain lookup"""
        if not self.virustotal_key:
            return None
        
        self.stats["vt_lookups"] += 1
        
        url = f"https://www.virustotal.com/api/v3/domains/{domain}"
        headers = {"x-apikey": self.virustotal_key}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get("data", {}).get("attributes", {})
            stats = data.get("last_analysis_stats", {})
            return {
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0)
            }
        
        return None
    
    def _check_virustotal_hash(self, file_hash: str) -> Optional[Dict]:
        """Query VirusTotal hash lookup"""
        if not self.virustotal_key:
            return None
        
        self.stats["vt_lookups"] += 1
        
        url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
        headers = {"x-apikey": self.virustotal_key}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get("data", {}).get("attributes", {})
            stats = data.get("last_analysis_stats", {})
            return {
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "total": sum(stats.values()) if stats else 0,
                "meaningful_name": data.get("meaningful_name", "")
            }
        
        return None
    
    def _get_cached(self, key: str) -> Optional[ThreatIntelResult]:
        """Get cached result"""
        if key in self._cache:
            result = self._cache[key]
            age = (datetime.now() - result.timestamp).total_seconds()
            if age < self.cache_ttl:
                self.stats["cache_hits"] += 1
                result.cached = True
                return result
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, result: ThreatIntelResult):
        """Cache result"""
        self._cache[key] = result
        
        # Limit cache size
        if len(self._cache) > 10000:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp
            )
            for k in sorted_keys[:1000]:
                del self._cache[k]
    
    def add_ioc(self, indicator: str, indicator_type: str, 
                threat: str = "User defined", confidence: float = 0.8):
        """Add indicator to local IOC database"""
        self.ioc_db[indicator.lower()] = {
            "type": indicator_type,
            "threat": threat,
            "confidence": confidence
        }
    
    def get_stats(self) -> Dict:
        """Get API statistics"""
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "ioc_db_size": len(self.ioc_db)
        }


# Singleton
_threat_intel = None

def get_threat_intel(virustotal_key: str = None, 
                     abuseipdb_key: str = None) -> ThreatIntelAPI:
    global _threat_intel
    if _threat_intel is None:
        _threat_intel = ThreatIntelAPI(virustotal_key, abuseipdb_key)
    return _threat_intel


if __name__ == "__main__":
    api = ThreatIntelAPI()
    
    print("\n🔍 Threat Intel API Test\n")
    
    # Test IP lookup
    test_ip = "185.220.101.1"
    print(f"Checking IP: {test_ip}")
    result = api.check_ip(test_ip)
    print(f"  Malicious: {result.is_malicious}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Sources: {result.sources}")
    
    # Test hash lookup
    test_hash = "44d88612fea8a8f36de82e1278abb02f"
    print(f"\nChecking hash: {test_hash}")
    result = api.check_hash(test_hash)
    print(f"  Malicious: {result.is_malicious}")
    print(f"  Confidence: {result.confidence}")
    
    print(f"\n📊 Stats: {api.get_stats()}")
