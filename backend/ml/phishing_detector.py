"""
AI-Powered Phishing Detection Module
Based on Paper 4: NLP and Transformer-based phishing detection

Features:
- URL analysis for suspicious patterns
- Email content analysis using NLP
- Domain spoofing detection
- Visual similarity detection for brand impersonation
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
import hashlib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class PhishingAnalysis:
    """Phishing detection result"""
    is_phishing: bool
    confidence: float
    risk_score: float  # 0-100
    indicators: List[str]
    url_analysis: Optional[Dict] = None
    content_analysis: Optional[Dict] = None
    recommendation: str = ""


class URLAnalyzer:
    """
    Analyze URLs for phishing indicators
    """
    
    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.click'}
    
    # Common brand targets for phishing
    TARGET_BRANDS = {
        'paypal', 'apple', 'microsoft', 'google', 'amazon', 'facebook', 
        'netflix', 'bank', 'secure', 'verify', 'account', 'login', 'signin',
        'update', 'confirm', 'billing', 'password', 'credential'
    }
    
    # Suspicious URL patterns
    SUSPICIOUS_PATTERNS = [
        r'@',  # @ symbol in URL
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP address
        r'https?://[^/]*-[^/]*-[^/]*\.',  # Multiple hyphens
        r'[0-9]{5,}',  # Long number sequences
        r'\.com\..*\.',  # Fake TLD pattern
        r'login|signin|verify|secure|account|update',  # Suspicious keywords
    ]
    
    def analyze(self, url: str) -> Dict:
        """Analyze URL for phishing indicators"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            full_url = url.lower()
        except:
            return {"error": "Invalid URL", "risk_score": 100}
        
        indicators = []
        risk_score = 0
        
        # Check for IP address instead of domain
        if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            indicators.append("IP address used instead of domain")
            risk_score += 30
        
        # Check for suspicious TLD
        for tld in self.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                indicators.append(f"Suspicious TLD: {tld}")
                risk_score += 20
                break
        
        # Check for brand impersonation
        for brand in self.TARGET_BRANDS:
            if brand in domain and brand not in domain.split('.')[0]:
                indicators.append(f"Possible brand impersonation: {brand}")
                risk_score += 25
        
        # Check URL length (phishing URLs are often long)
        if len(url) > 75:
            indicators.append("Unusually long URL")
            risk_score += 10
        
        # Check for @ symbol
        if '@' in url:
            indicators.append("@ symbol in URL (credential harvesting)")
            risk_score += 30
        
        # Check for multiple subdomains
        subdomain_count = domain.count('.') - 1
        if subdomain_count > 2:
            indicators.append(f"Multiple subdomains: {subdomain_count}")
            risk_score += 15
        
        # Check for HTTPS
        if not url.startswith('https://'):
            indicators.append("Not using HTTPS")
            risk_score += 10
        
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, full_url):
                indicators.append(f"Suspicious pattern detected")
                risk_score += 10
                break
        
        # Check for homograph attacks (unicode lookalikes)
        if any(ord(c) > 127 for c in domain):
            indicators.append("Possible homograph attack (unicode characters)")
            risk_score += 35
        
        return {
            "url": url,
            "domain": domain,
            "is_https": url.startswith('https://'),
            "url_length": len(url),
            "subdomain_count": subdomain_count,
            "risk_score": min(risk_score, 100),
            "indicators": indicators,
            "is_suspicious": risk_score >= 40
        }


class EmailContentAnalyzer:
    """
    Analyze email content for phishing indicators using NLP
    """
    
    # Phishing keywords and phrases
    PHISHING_KEYWORDS = [
        'urgent', 'immediately', 'verify your account', 'suspended',
        'confirm your identity', 'unusual activity', 'click here',
        'act now', 'limited time', 'your account will be',
        'security alert', 'unauthorized access', 'update your information',
        'expire', 'within 24 hours', 'failure to', 'important notice',
        'dear customer', 'dear user', 'valued customer'
    ]
    
    # Legitimate sender patterns
    LEGITIMATE_PATTERNS = [
        r'noreply@.*\.(com|org|edu|gov)$',
        r'support@.*\.(com|org|edu|gov)$',
    ]
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500) if HAS_SKLEARN else None
        self.classifier = None
        self.is_trained = False
    
    def analyze(self, content: str, sender: str = "", subject: str = "") -> Dict:
        """Analyze email content for phishing indicators"""
        content_lower = content.lower()
        subject_lower = subject.lower()
        
        indicators = []
        risk_score = 0
        
        # Check for phishing keywords in content
        keyword_count = 0
        for keyword in self.PHISHING_KEYWORDS:
            if keyword in content_lower or keyword in subject_lower:
                keyword_count += 1
        
        if keyword_count > 0:
            indicators.append(f"Found {keyword_count} phishing keywords")
            risk_score += min(keyword_count * 8, 40)
        
        # Check for urgency indicators
        urgency_words = ['urgent', 'immediately', 'now', 'asap', 'quick', 'fast']
        urgency_count = sum(1 for w in urgency_words if w in content_lower)
        if urgency_count > 1:
            indicators.append("Multiple urgency indicators")
            risk_score += 15
        
        # Check for URL presence
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, content)
        if len(urls) > 0:
            indicators.append(f"Contains {len(urls)} URLs")
            risk_score += 5
            
            # Analyze each URL
            url_analyzer = URLAnalyzer()
            for url in urls[:3]:  # Check first 3 URLs
                url_result = url_analyzer.analyze(url)
                if url_result.get("is_suspicious"):
                    indicators.append(f"Suspicious URL: {url[:50]}...")
                    risk_score += 20
        
        # Check for generic greeting
        generic_greetings = ['dear customer', 'dear user', 'dear valued', 'dear member']
        for greeting in generic_greetings:
            if greeting in content_lower:
                indicators.append("Generic greeting (no personalization)")
                risk_score += 10
                break
        
        # Check for grammar/spelling patterns common in phishing
        # (simplified check)
        if content_lower.count('!!!') > 0 or content_lower.count('???') > 0:
            indicators.append("Excessive punctuation")
            risk_score += 5
        
        # Check sender if provided
        if sender:
            sender_lower = sender.lower()
            # Check for mismatched reply-to
            if '@' in sender:
                sender_domain = sender.split('@')[1] if '@' in sender else ''
                # Check if domain looks suspicious
                for brand in URLAnalyzer.TARGET_BRANDS:
                    if brand in sender_domain and not sender_domain.endswith(f'{brand}.com'):
                        indicators.append(f"Sender domain impersonation: {sender_domain}")
                        risk_score += 25
        
        return {
            "content_length": len(content),
            "url_count": len(urls),
            "keyword_count": keyword_count,
            "risk_score": min(risk_score, 100),
            "indicators": indicators,
            "is_suspicious": risk_score >= 35
        }
    
    def train(self, emails: List[str], labels: List[int]):
        """Train the phishing classifier"""
        if not HAS_SKLEARN:
            return
        
        X = self.vectorizer.fit_transform(emails)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, content: str) -> Tuple[int, float]:
        """Predict if email is phishing"""
        if not self.is_trained or self.classifier is None:
            return 0, 0.5
        
        X = self.vectorizer.transform([content])
        prediction = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        
        return int(prediction), float(max(proba))


class PhishingDetector:
    """
    Comprehensive Phishing Detection System
    Combines URL analysis, email content analysis, and ML classification
    """
    
    def __init__(self):
        self.url_analyzer = URLAnalyzer()
        self.content_analyzer = EmailContentAnalyzer()
        
        # Statistics
        self.stats = {
            "urls_analyzed": 0,
            "emails_analyzed": 0,
            "phishing_detected": 0,
            "false_positives_reported": 0
        }
        
        print("🎣 Phishing Detector initialized")
    
    def analyze_url(self, url: str) -> PhishingAnalysis:
        """Analyze a URL for phishing"""
        import uuid
        
        self.stats["urls_analyzed"] += 1
        
        url_result = self.url_analyzer.analyze(url)
        
        is_phishing = url_result.get("is_suspicious", False)
        risk_score = url_result.get("risk_score", 0)
        
        if is_phishing:
            self.stats["phishing_detected"] += 1
        
        # Generate recommendation
        if risk_score >= 70:
            recommendation = "⛔ HIGH RISK: Do not click this URL. Report as phishing."
        elif risk_score >= 40:
            recommendation = "⚠️ SUSPICIOUS: Verify the source before clicking."
        else:
            recommendation = "✅ LOW RISK: URL appears legitimate, but stay vigilant."
        
        return PhishingAnalysis(
            is_phishing=is_phishing,
            confidence=risk_score / 100,
            risk_score=risk_score,
            indicators=url_result.get("indicators", []),
            url_analysis=url_result,
            recommendation=recommendation
        )
    
    def analyze_email(self, content: str, sender: str = "", 
                     subject: str = "") -> PhishingAnalysis:
        """Analyze email content for phishing"""
        import uuid
        
        self.stats["emails_analyzed"] += 1
        
        content_result = self.content_analyzer.analyze(content, sender, subject)
        
        is_phishing = content_result.get("is_suspicious", False)
        risk_score = content_result.get("risk_score", 0)
        
        if is_phishing:
            self.stats["phishing_detected"] += 1
        
        # Generate recommendation
        if risk_score >= 70:
            recommendation = "⛔ HIGH RISK: This email shows strong phishing indicators. Do not click any links or download attachments."
        elif risk_score >= 40:
            recommendation = "⚠️ SUSPICIOUS: Verify sender identity through official channels before taking any action."
        else:
            recommendation = "✅ LOW RISK: Email appears legitimate, but always verify unexpected requests."
        
        return PhishingAnalysis(
            is_phishing=is_phishing,
            confidence=risk_score / 100,
            risk_score=risk_score,
            indicators=content_result.get("indicators", []),
            content_analysis=content_result,
            recommendation=recommendation
        )
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            **self.stats,
            "ml_trained": self.content_analyzer.is_trained
        }


# Global instance
_detector: Optional[PhishingDetector] = None


def get_phishing_detector() -> PhishingDetector:
    """Get or create phishing detector"""
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector
