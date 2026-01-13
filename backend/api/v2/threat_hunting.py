"""
Threat Hunting API
CrowdStrike-style query-based threat hunting

Features:
- Search across all telemetry
- IOC lookup integration
- Saved queries library
- Timeline correlation
- Query language (simplified SPL)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
from collections import defaultdict


router = APIRouter(prefix="/hunt", tags=["Threat Hunting"])


# In-memory storage for hunting
telemetry_store: List[Dict] = []
saved_queries: Dict[str, Dict] = {}
hunt_results_cache: Dict[str, Dict] = {}
ioc_database: Dict[str, Dict] = {}


class TelemetryEvent(BaseModel):
    """Telemetry event for hunting"""
    timestamp: str
    event_type: str  # process, network, file, registry, dns
    source: str
    data: Dict


class HuntQuery(BaseModel):
    """Hunt query request"""
    query: str  # Simplified query language
    time_range_hours: int = 24
    limit: int = 100


class SavedQuery(BaseModel):
    """Saved hunt query"""
    name: str
    description: str
    query: str
    tags: List[str] = []


class IOCLookup(BaseModel):
    """IOC lookup request"""
    indicator: str
    indicator_type: str  # ip, domain, hash, url


# Initialize with sample IOCs
def init_ioc_database():
    """Initialize sample IOC database"""
    sample_iocs = {
        # Known malicious IPs
        "185.220.101.1": {"type": "ip", "threat": "C2 Server", "confidence": 0.95},
        "45.77.65.211": {"type": "ip", "threat": "Tor Exit Node", "confidence": 0.85},
        "192.168.1.100": {"type": "ip", "threat": "Internal Scanner", "confidence": 0.60},
        
        # Known malicious domains
        "evil.com": {"type": "domain", "threat": "Phishing", "confidence": 0.90},
        "malware-c2.net": {"type": "domain", "threat": "C2 Domain", "confidence": 0.95},
        
        # Known malicious hashes
        "a1b2c3d4e5f6": {"type": "hash", "threat": "Ransomware", "confidence": 0.99},
        "deadbeef1234": {"type": "hash", "threat": "Trojan", "confidence": 0.95},
    }
    
    for indicator, info in sample_iocs.items():
        ioc_database[indicator.lower()] = {
            "indicator": indicator,
            **info,
            "first_seen": "2024-01-01",
            "last_seen": datetime.now().isoformat()
        }


init_ioc_database()


# Initialize with sample saved queries (CrowdStrike-style)
def init_saved_queries():
    """Initialize sample saved queries"""
    queries = [
        {
            "name": "PowerShell Execution",
            "description": "Find all PowerShell process executions",
            "query": "event_type:process AND process_name:powershell",
            "tags": ["execution", "lolbas", "T1059"]
        },
        {
            "name": "Suspicious Network Connections",
            "description": "External connections on non-standard ports",
            "query": "event_type:network AND port:>1024 AND NOT port:443 AND NOT port:80",
            "tags": ["network", "c2", "exfiltration"]
        },
        {
            "name": "Registry Persistence",
            "description": "Registry modifications in Run keys",
            "query": "event_type:registry AND path:*\\Run\\*",
            "tags": ["persistence", "T1547"]
        },
        {
            "name": "Lateral Movement",
            "description": "SMB/RDP connections to internal hosts",
            "query": "event_type:network AND (port:445 OR port:3389) AND dst_ip:192.168.*",
            "tags": ["lateral", "T1021"]
        },
        {
            "name": "High Entropy File Writes",
            "description": "Potential ransomware file encryption",
            "query": "event_type:file AND action:write AND entropy:>7.5",
            "tags": ["ransomware", "impact"]
        },
        {
            "name": "Credential Access Tools",
            "description": "Mimikatz and similar tools",
            "query": "event_type:process AND (process_name:*mimikatz* OR cmdline:*sekurlsa*)",
            "tags": ["credential", "T1003"]
        },
        {
            "name": "DNS Tunneling",
            "description": "Unusually long DNS queries",
            "query": "event_type:dns AND query_length:>50",
            "tags": ["exfiltration", "c2", "T1071"]
        },
        {
            "name": "Process Injection",
            "description": "CreateRemoteThread and similar",
            "query": "event_type:process AND (action:inject OR api:CreateRemoteThread)",
            "tags": ["injection", "T1055"]
        },
    ]
    
    for q in queries:
        saved_queries[q["name"]] = {
            **q,
            "created_at": datetime.now().isoformat(),
            "run_count": 0
        }


init_saved_queries()


def parse_query(query: str, events: List[Dict]) -> List[Dict]:
    """
    Parse and execute simplified query language
    
    Supports:
    - field:value (exact match)
    - field:*value* (wildcard)
    - field:>N, field:<N (numeric comparison)
    - AND, OR, NOT
    """
    if not query.strip():
        return events
    
    results = []
    
    # Split by AND/OR
    and_parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
    
    for event in events:
        match = True
        
        for part in and_parts:
            # Handle NOT
            is_negated = False
            if part.strip().upper().startswith("NOT "):
                is_negated = True
                part = part[4:].strip()
            
            # Parse field:value
            if ":" in part:
                field, value = part.split(":", 1)
                field = field.strip().lower()
                value = value.strip()
                
                # Get event value (check data dict too)
                event_value = event.get(field) or event.get("data", {}).get(field)
                
                if event_value is None:
                    part_match = False
                elif value.startswith(">"):
                    try:
                        part_match = float(event_value) > float(value[1:])
                    except:
                        part_match = False
                elif value.startswith("<"):
                    try:
                        part_match = float(event_value) < float(value[1:])
                    except:
                        part_match = False
                elif "*" in value:
                    # Wildcard match
                    pattern = value.replace("*", ".*")
                    part_match = bool(re.search(pattern, str(event_value), re.IGNORECASE))
                else:
                    # Exact match
                    part_match = str(event_value).lower() == value.lower()
                
                if is_negated:
                    part_match = not part_match
                
                if not part_match:
                    match = False
                    break
        
        if match:
            results.append(event)
    
    return results


@router.post("/query")
async def execute_hunt_query(request: HuntQuery) -> Dict:
    """
    Execute a hunt query across telemetry
    
    Query syntax:
    - field:value - Exact match
    - field:*pattern* - Wildcard match
    - field:>N - Greater than
    - field1:val1 AND field2:val2 - Multiple conditions
    - NOT field:value - Negation
    """
    # Filter by time range
    cutoff = datetime.now() - timedelta(hours=request.time_range_hours)
    recent_events = [
        e for e in telemetry_store
        if datetime.fromisoformat(e.get("timestamp", "2000-01-01")) > cutoff
    ]
    
    # Execute query
    results = parse_query(request.query, recent_events)
    
    # Limit results
    results = results[:request.limit]
    
    # Cache results
    import uuid
    result_id = str(uuid.uuid4())[:8]
    hunt_results_cache[result_id] = {
        "query": request.query,
        "results": results,
        "count": len(results),
        "executed_at": datetime.now().isoformat()
    }
    
    return {
        "result_id": result_id,
        "query": request.query,
        "results": results,
        "count": len(results),
        "time_range_hours": request.time_range_hours,
        "executed_at": datetime.now().isoformat()
    }


@router.get("/saved-queries")
async def get_saved_queries() -> Dict:
    """Get all saved hunt queries"""
    return {
        "queries": list(saved_queries.values()),
        "count": len(saved_queries)
    }


@router.post("/saved-queries")
async def create_saved_query(query: SavedQuery) -> Dict:
    """Save a new hunt query"""
    saved_queries[query.name] = {
        "name": query.name,
        "description": query.description,
        "query": query.query,
        "tags": query.tags,
        "created_at": datetime.now().isoformat(),
        "run_count": 0
    }
    
    return {"saved": True, "name": query.name}


@router.delete("/saved-queries/{name}")
async def delete_saved_query(name: str) -> Dict:
    """Delete a saved query"""
    if name in saved_queries:
        del saved_queries[name]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Query not found")


@router.post("/ioc/lookup")
async def lookup_ioc(request: IOCLookup) -> Dict:
    """
    Look up an indicator of compromise
    
    Checks:
    - Local IOC database
    - Returns threat info if found
    """
    indicator = request.indicator.lower().strip()
    
    # Check local database
    if indicator in ioc_database:
        ioc = ioc_database[indicator]
        return {
            "found": True,
            "indicator": ioc["indicator"],
            "type": ioc["type"],
            "threat": ioc["threat"],
            "confidence": ioc["confidence"],
            "first_seen": ioc.get("first_seen"),
            "last_seen": ioc.get("last_seen"),
            "source": "local_database"
        }
    
    # Not found
    return {
        "found": False,
        "indicator": request.indicator,
        "type": request.indicator_type,
        "message": "Indicator not found in database"
    }


@router.post("/ioc/add")
async def add_ioc(
    indicator: str,
    indicator_type: str,
    threat: str,
    confidence: float = 0.8
) -> Dict:
    """Add a new IOC to the database"""
    ioc_database[indicator.lower()] = {
        "indicator": indicator,
        "type": indicator_type,
        "threat": threat,
        "confidence": confidence,
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat()
    }
    
    return {"added": True, "indicator": indicator}


@router.get("/ioc/list")
async def list_iocs(
    ioc_type: Optional[str] = None,
    limit: int = 100
) -> Dict:
    """List all IOCs in database"""
    iocs = list(ioc_database.values())
    
    if ioc_type:
        iocs = [i for i in iocs if i["type"] == ioc_type]
    
    return {
        "iocs": iocs[:limit],
        "count": len(iocs),
        "types": list(set(i["type"] for i in ioc_database.values()))
    }


@router.post("/telemetry")
async def ingest_telemetry(event: TelemetryEvent) -> Dict:
    """
    Ingest telemetry event for hunting
    
    Events are stored in memory for querying
    """
    event_dict = {
        "timestamp": event.timestamp,
        "event_type": event.event_type,
        "source": event.source,
        **event.data
    }
    
    telemetry_store.append(event_dict)
    
    # Keep last 10000 events
    if len(telemetry_store) > 10000:
        telemetry_store.pop(0)
    
    # Check against IOCs
    ioc_hits = []
    for field, value in event.data.items():
        if str(value).lower() in ioc_database:
            ioc_hits.append({
                "field": field,
                "value": value,
                "ioc": ioc_database[str(value).lower()]
            })
    
    return {
        "ingested": True,
        "ioc_hits": ioc_hits,
        "total_events": len(telemetry_store)
    }


@router.get("/timeline")
async def get_hunt_timeline(
    hours: int = 24,
    event_type: Optional[str] = None
) -> Dict:
    """
    Get timeline of events for correlation
    
    Groups events by hour for visualization
    """
    cutoff = datetime.now() - timedelta(hours=hours)
    
    # Filter events
    events = [
        e for e in telemetry_store
        if datetime.fromisoformat(e.get("timestamp", "2000-01-01")) > cutoff
    ]
    
    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]
    
    # Group by hour
    hourly = defaultdict(list)
    for event in events:
        try:
            dt = datetime.fromisoformat(event["timestamp"])
            hour_key = dt.strftime("%Y-%m-%d %H:00")
            hourly[hour_key].append(event)
        except:
            pass
    
    timeline = [
        {
            "hour": hour,
            "count": len(events),
            "types": list(set(e.get("event_type", "unknown") for e in events))
        }
        for hour, events in sorted(hourly.items())
    ]
    
    return {
        "timeline": timeline,
        "total_events": len(events),
        "hours": hours,
        "event_types": list(set(e.get("event_type", "unknown") for e in events))
    }


@router.post("/simulate")
async def simulate_events(count: int = 100) -> Dict:
    """Generate simulated telemetry for testing"""
    import random
    
    event_types = ["process", "network", "file", "registry", "dns"]
    process_names = ["powershell.exe", "cmd.exe", "notepad.exe", "chrome.exe", "explorer.exe",
                    "svchost.exe", "mimikatz.exe", "psexec.exe", "wscript.exe"]
    
    generated = 0
    for _ in range(count):
        event_type = random.choice(event_types)
        
        if event_type == "process":
            data = {
                "process_name": random.choice(process_names),
                "pid": random.randint(1000, 65535),
                "cmdline": f"-encoded {random.randbytes(20).hex()}" if random.random() > 0.7 else ""
            }
        elif event_type == "network":
            data = {
                "src_ip": f"192.168.1.{random.randint(1, 254)}",
                "dst_ip": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" if random.random() > 0.5 else f"192.168.1.{random.randint(1, 254)}",
                "port": random.choice([80, 443, 445, 3389, 4444, 8080, random.randint(10000, 65535)])
            }
        elif event_type == "file":
            data = {
                "path": f"C:\\Users\\test\\{random.choice(['Documents', 'Downloads', 'Desktop'])}\\file{random.randint(1, 100)}.{random.choice(['exe', 'dll', 'txt', 'pdf'])}",
                "action": random.choice(["read", "write", "delete"]),
                "entropy": round(random.uniform(3.0, 8.0), 2)
            }
        elif event_type == "registry":
            data = {
                "path": random.choice([
                    "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\malware",
                    "HKCU\\Software\\test",
                    "HKLM\\SYSTEM\\Services\\testsvc"
                ]),
                "action": random.choice(["create", "modify", "delete"])
            }
        else:  # dns
            data = {
                "query": f"{random.choice(['www', 'api', 'mail'])}.{random.choice(['google.com', 'evil.com', 'malware-c2.net', 'example.com'])}",
                "query_length": random.randint(10, 100)
            }
        
        telemetry_store.append({
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat(),
            "event_type": event_type,
            "source": "simulation",
            **data
        })
        generated += 1
    
    return {
        "generated": generated,
        "total_events": len(telemetry_store)
    }


@router.get("/stats")
async def get_hunt_stats() -> Dict:
    """Get hunting statistics"""
    type_counts = defaultdict(int)
    for event in telemetry_store:
        type_counts[event.get("event_type", "unknown")] += 1
    
    return {
        "total_events": len(telemetry_store),
        "event_types": dict(type_counts),
        "saved_queries": len(saved_queries),
        "iocs": len(ioc_database),
        "cached_results": len(hunt_results_cache)
    }
