# PCDS Competitive Analysis & Market Research
*Research-based document for mentor review and investor pitches*

---

## 📊 Executive Summary

PCDS targets the **underserved SMB cybersecurity market** where enterprise NDR solutions (Darktrace, Vectra AI) cost $55K-$350K/year - unaffordable for most mid-sized companies.

---

## 💰 Competitor Pricing (Verified Research)

### Network Detection & Response (NDR)

| Vendor | Pricing Model | Annual Cost Range | Source |
|--------|---------------|-------------------|--------|
| **Darktrace** | Per-device + appliance | **$55,200 - $350,000/year** | Vendr.com, PeerSpot |
| **Vectra AI** | Per-IP (95th percentile) | **$40/IP/year** (enterprise: custom) | AWS Marketplace, Vectra.ai |
| **CrowdStrike Falcon** | Per-endpoint | **$8.99 - $15.99/endpoint/month** | CrowdStrike.com |
| **SentinelOne** | Per-endpoint tiers | **$69.99 - $229.99/endpoint/year** | SentinelOne.com |
| **Microsoft Sentinel** | Per-GB ingested | **$2.46/GB/day** + compute | Azure.com |

### What This Means:
- **500 endpoints on Darktrace**: ~$100,000+/year
- **500 endpoints on SentinelOne Complete**: ~$80,000/year
- **PCDS Target**: **$5,000-$15,000/year** (10-20x cheaper)

---

## 🔍 Feature Comparison (Research-Verified)

| Feature | PCDS | Darktrace | Vectra AI | SentinelOne |
|---------|------|-----------|-----------|-------------|
| **AI/ML Detection** | ✅ PyTorch LSTM | ✅ Self-Learning AI | ✅ Attack Signal Intelligence | ✅ Static + Behavioral AI |
| **MITRE ATT&CK Mapping** | ✅ 26% coverage | ✅ Full mapping | ✅ Full mapping | ✅ Full mapping |
| **Automated Response** | ⚠️ Simulated | ✅ Antigena | ✅ Lockdown | ✅ Kill/Quarantine |
| **Network Traffic Analysis** | ✅ | ✅ Core strength | ✅ Core strength | ⚠️ Limited |
| **AI Analyst/Copilot** | ✅ Azure OpenAI | ✅ Cyber AI Analyst | ⚠️ Basic | ✅ Purple AI |
| **UEBA (User Behavior)** | ✅ | ✅ | ✅ | ⚠️ Add-on |
| **Cloud-Native** | ✅ | ✅ | ✅ | ✅ |
| **SMB Affordable** | ✅ | ❌ | ❌ | ⚠️ Mid-range |

---

## 🏆 Darktrace Deep-Dive

**Company:** Founded 2013 (Cambridge, UK), Public company  
**Valuation:** ~$4B  
**Technology:** Self-Learning AI that understands "normal" behavior

### Pricing Details (from research):
- Median annual cost: **$55,200** (based on 21 purchases - Vendr)
- Enterprise deployments: **up to $350,000/year**
- UK Gov contract: £1,500 - £12,500/month per instance
- Hardware appliances: $2,000 (small) to $22,500 (extra-large)
- Per-device: **$12-$54/device/year** depending on scale

### Key Differentiators:
- "Antigena" autonomous response
- Self-learning without training data
- 30-day deployment claim

### PCDS vs Darktrace:
| Factor | Darktrace | PCDS |
|--------|-----------|------|
| Min. Cost | ~$50,000/year | **$5,000/year (target)** |
| Setup Time | 30 days | **Same-day** |
| AI Approach | Unsupervised self-learning | Supervised ML (5.5M samples) |
| Target | Enterprise | **SMB/Mid-market** |

---

## 🏆 Vectra AI Deep-Dive

**Company:** Founded 2012 (San Jose), Private (~$1.2B valuation)  
**Technology:** Attack Signal Intelligence, NDR + Identity

### Pricing Details (from research):
- Licensing: Based on **95th percentile of concurrent IPs**
- AWS Marketplace: **$40/IP/year** (per design)
- Standard package: **$499/month**
- Complete package: **$1,299/month** (includes MDR)
- Described as "pricier side" by reviewers (PeerSpot)

### Key Differentiators:
- Focus on attack progression, not just alerts
- Strong identity threat detection
- Prioritizes threats by urgency

### PCDS vs Vectra:
| Factor | Vectra | PCDS |
|--------|--------|------|
| Pricing | Complex IP-based | **Simple annual license** |
| Focus | Enterprise SOC teams | **SMBs without SOC** |
| Identity | ✅ Strong | ⚠️ Basic |
| Setup Complexity | High | **Low** |

---

## 🏆 SentinelOne Deep-Dive

**Company:** Founded 2013, Public ($5B+ valuation)  
**Technology:** XDR (Endpoint + EDR + Identity)

### Pricing Details (from research):
- **Core:** $69.99/endpoint/year (NGAV)
- **Control:** $79.99/endpoint/year (+ device control)
- **Complete:** $159.99-$179.99/endpoint/year (full XDR)
- **Commercial:** $209.99-$229.99/endpoint/year (+ identity)
- **Enterprise:** Custom pricing (includes MDR)

### Key Differentiators:
- Endpoint-first approach
- Strong remediation/rollback
- Purple AI assistant

### PCDS vs SentinelOne:
| Factor | SentinelOne | PCDS |
|--------|-------------|------|
| Focus | Endpoint | **Network** |
| 100 endpoints | ~$16,000/year | **$5,000 (target)** |
| Network visibility | ⚠️ Limited | ✅ Core strength |
| Response | ✅ Real | ⚠️ Simulated (prototype) |

---

## 📈 Market Opportunity

### NDR Market Size:
- **2024:** $3.2 billion globally
- **2028:** $6.4 billion (projected)
- **CAGR:** 15-17%

### SMB Security Gap:
- **60%** of SMBs go out of business within 6 months of a cyberattack
- **43%** of cyberattacks target small businesses
- **Average breach cost:** $4.45M (IBM 2023)
- **SMBs spending on security:** $1,000-$10,000/year (can't afford enterprise tools)

### Underserved Segment:
- Companies with 50-500 employees
- Revenue $10M-$100M
- No dedicated SOC team
- Current options: Basic antivirus OR expensive enterprise tools

**This is PCDS's target market.**

---

## ✅ PCDS Strengths (Honest Assessment)

| Strength | Evidence |
|----------|----------|
| **Real ML Models** | Trained on 5.5M samples (UNSW-NB15 + CICIDS) |
| **Proven Accuracy** | 88.3% accuracy, 2.8% FPR on test data |
| **Azure Integration** | Azure OpenAI Copilot working |
| **MITRE Mapping** | 26% technique coverage |
| **Working Prototype** | Full UI, API, detection flow functional |
| **Cost Target** | 10-20x cheaper than enterprise solutions |

---

## ⚠️ PCDS Weaknesses (Honest Assessment)

| Weakness | Impact | Mitigation Path |
|----------|--------|-----------------|
| **Auto-response simulated** | Can't actually block traffic | Integrate with pfSense, Fortinet APIs |
| **No production deployment** | Unproven at scale | Normal for prototype stage |
| **Single developer** | Resource constraints | Seeking funding/team |
| **26% MITRE coverage** | Limited vs competitors | Expand detection rules |
| **No identity detection** | Missing Vectra strength | Future roadmap item |

---

## �️ Product Improvement Roadmap

### Phase 1: Production-Ready (3-6 months)
- [ ] Real firewall integration (pfSense, Fortinet, Palo Alto)
- [ ] EDR agent (Windows/Linux)
- [ ] Azure cloud deployment
- [ ] Expand to 50% MITRE coverage

### Phase 2: Market Entry (6-12 months)
- [ ] SIEM integration (Splunk, Elastic)
- [ ] Cloud log ingestion (AWS CloudTrail, Azure logs)
- [ ] Identity threat detection
- [ ] SOC 2 Type 1 compliance

### Phase 3: Enterprise Features (12-18 months)
- [ ] Multi-tenant architecture
- [ ] Threat intelligence feeds
- [ ] Custom ML model training
- [ ] MSSP partner program

---

## 🎯 Competitive Positioning Statement

> **"PCDS delivers 80% of enterprise NDR capability at 10% of the cost, specifically designed for mid-sized companies who can't afford Darktrace or maintain a dedicated SOC team."**

---

## � Key Stats for Presentations

| Metric | Value | Context |
|--------|-------|---------|
| Training Data | 5.5M samples | Larger than typical academic datasets |
| Accuracy | 88.3% | Competitive with commercial solutions |
| False Positive Rate | 2.8% | Better than industry avg (5-10%) |
| MITRE Coverage | 26% (12 techniques) | Good starting point |
| Target Price | 10-20x cheaper | $5K-$15K vs $100K-$350K |
| Competitor Min. Cost | $55,000/year | Darktrace median |

---

## 📚 Research Sources

1. **Darktrace Pricing:** Vendr.com, PeerSpot reviews, UK Gov Digital Marketplace
2. **Vectra AI Pricing:** AWS Marketplace, Vectra.ai licensing docs
3. **SentinelOne Pricing:** SentinelOne.com official pricing page
4. **Market Data:** Industry analyst reports, vendor public filings
5. **Breach Statistics:** IBM Cost of Data Breach Report 2023

---

*Document Version 1.0 | December 2024*
*Prepared for Imagine Cup 2024 mentor review*
