"""
Generate Realistic Network Traffic Data for PCDS ML Training
Creates 1M records with realistic attack patterns
"""
import pandas as pd
import numpy as np
import time

def generate_realistic_data(n=1000000):
    print(f"🔄 Generating {n:,} realistic network events...")
    start_time = time.time()
    
    # Realistic IP ranges
    internal_ips = [f"192.168.1.{i}" for i in range(1, 255)]
    server_ips = [f"10.0.0.{i}" for i in range(1, 50)]
    external_ips = [
        "8.8.8.8", "1.1.1.1", "13.107.42.14", "52.84.12.33",  # Google, Cloudflare, Microsoft
        "151.101.1.69", "140.82.114.4", "172.217.14.78"  # Reddit, GitHub, Google
    ]
    suspicious_ips = [
        "185.234.219.10", "91.121.87.10", "23.94.5.133",  # Known C2
        "45.33.32.156", "194.26.192.71", "193.233.232.119"
    ]
    
    # Realistic processes
    normal_procs = ["chrome.exe", "firefox.exe", "outlook.exe", "teams.exe", 
                    "edge.exe", "slack.exe", "zoom.exe", "code.exe"]
    suspicious_procs = ["powershell.exe", "cmd.exe", "wmic.exe", "psexec.exe", 
                        "mimikatz.exe", "nc.exe", "certutil.exe", "bitsadmin.exe"]
    
    # Attack-specific ports
    normal_ports = [80, 443, 8080, 8443]
    recon_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080]
    lateral_ports = [22, 135, 139, 445, 3389, 5985, 5986]  # SSH, RPC, SMB, RDP, WinRM
    credential_ports = [88, 389, 636, 445, 464]  # Kerberos, LDAP, SMB
    c2_ports = [443, 8443, 4444, 1337, 6666, 9999]  # Common C2 ports
    
    # Labels with realistic distribution
    labels = np.random.choice(
        ["benign", "recon", "lateral_movement", "credential_access", 
         "exfiltration", "c2_communication", "privilege_escalation"],
        n, 
        p=[0.82, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02]
    )
    
    # Base data
    data = {
        "timestamp": pd.date_range(start="2026-01-23", periods=n, freq='ms'),
        "src_ip": np.random.choice(internal_ips, n),
        "dst_ip": np.random.choice(external_ips + server_ips, n),
        "src_port": np.random.randint(49152, 65535, n),
        "dst_port": np.random.choice(normal_ports, n),
        "protocol": np.random.choice(["TCP", "UDP"], n, p=[0.92, 0.08]),
        "bytes_sent": np.random.gamma(shape=2, scale=500, size=n).astype(int),
        "bytes_recv": np.random.gamma(shape=5, scale=2000, size=n).astype(int),
        "packets": np.random.poisson(lam=10, size=n),
        "duration_ms": np.random.exponential(scale=200, size=n).astype(int),
        "process_name": np.random.choice(normal_procs, n),
        "status": np.random.choice(["ESTABLISHED", "SYN_SENT", "CLOSE_WAIT", "TIME_WAIT"], n, p=[0.7, 0.15, 0.1, 0.05]),
        "label": labels
    }
    
    df = pd.DataFrame(data)
    
    # ============ Make attacks realistic ============
    
    # RECON: Small packets, many ports, fast connections
    recon_mask = df['label'] == 'recon'
    df.loc[recon_mask, 'dst_port'] = np.random.choice(recon_ports, size=recon_mask.sum())
    df.loc[recon_mask, 'bytes_sent'] = np.random.randint(40, 200, size=recon_mask.sum())
    df.loc[recon_mask, 'bytes_recv'] = np.random.randint(0, 100, size=recon_mask.sum())
    df.loc[recon_mask, 'duration_ms'] = np.random.randint(1, 50, size=recon_mask.sum())
    df.loc[recon_mask, 'status'] = np.random.choice(["SYN_SENT", "CLOSE_WAIT"], size=recon_mask.sum())
    
    # LATERAL MOVEMENT: Internal to internal, RDP/SMB/SSH
    lateral_mask = df['label'] == 'lateral_movement'
    df.loc[lateral_mask, 'dst_ip'] = np.random.choice(internal_ips + server_ips, size=lateral_mask.sum())
    df.loc[lateral_mask, 'dst_port'] = np.random.choice(lateral_ports, size=lateral_mask.sum())
    df.loc[lateral_mask, 'process_name'] = np.random.choice(suspicious_procs[:4], size=lateral_mask.sum())
    
    # CREDENTIAL ACCESS: Kerberos, LDAP attacks
    cred_mask = df['label'] == 'credential_access'
    df.loc[cred_mask, 'dst_ip'] = np.random.choice(server_ips[:10], size=cred_mask.sum())  # Domain controllers
    df.loc[cred_mask, 'dst_port'] = np.random.choice(credential_ports, size=cred_mask.sum())
    df.loc[cred_mask, 'process_name'] = np.random.choice(suspicious_procs, size=cred_mask.sum())
    df.loc[cred_mask, 'bytes_recv'] = np.random.randint(5000, 50000, size=cred_mask.sum())  # Ticket responses
    
    # EXFILTRATION: Huge outbound, external suspicious IPs
    exfil_mask = df['label'] == 'exfiltration'
    df.loc[exfil_mask, 'dst_ip'] = np.random.choice(suspicious_ips, size=exfil_mask.sum())
    df.loc[exfil_mask, 'bytes_sent'] = np.random.randint(1_000_000, 50_000_000, size=exfil_mask.sum())
    df.loc[exfil_mask, 'dst_port'] = np.random.choice([443, 8443, 22], size=exfil_mask.sum())
    df.loc[exfil_mask, 'duration_ms'] = np.random.randint(5000, 60000, size=exfil_mask.sum())
    
    # C2 COMMUNICATION: Periodic beacons, encrypted ports
    c2_mask = df['label'] == 'c2_communication'
    df.loc[c2_mask, 'dst_ip'] = np.random.choice(suspicious_ips, size=c2_mask.sum())
    df.loc[c2_mask, 'dst_port'] = np.random.choice(c2_ports, size=c2_mask.sum())
    df.loc[c2_mask, 'bytes_sent'] = np.random.randint(50, 500, size=c2_mask.sum())  # Small beacons
    df.loc[c2_mask, 'bytes_recv'] = np.random.randint(100, 2000, size=c2_mask.sum())  # Commands
    df.loc[c2_mask, 'process_name'] = np.random.choice(["rundll32.exe", "regsvr32.exe", "svchost.exe"], size=c2_mask.sum())
    
    # PRIVILEGE ESCALATION: Local processes, specific ports
    privesc_mask = df['label'] == 'privilege_escalation'
    df.loc[privesc_mask, 'dst_ip'] = "127.0.0.1"
    df.loc[privesc_mask, 'dst_port'] = np.random.choice([135, 445, 5985], size=privesc_mask.sum())
    df.loc[privesc_mask, 'process_name'] = np.random.choice(suspicious_procs, size=privesc_mask.sum())
    
    # Save to CSV
    df.to_csv("data/pcds_realistic_training.csv", index=False)
    
    elapsed = round(time.time() - start_time, 2)
    print(f"✅ Generated {n:,} records in {elapsed} seconds")
    print(f"\n📊 Label Distribution:")
    print(df['label'].value_counts())
    print(f"\n💾 Saved to: data/pcds_realistic_training.csv")
    
    return df

if __name__ == "__main__":
    generate_realistic_data(1_000_000)
