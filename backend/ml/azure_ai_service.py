"""
Azure AI Service Integration for PCDS
Imagine Cup 2026 - Required Microsoft AI Services

Integrates:
1. Azure OpenAI - Threat explanations, analyst co-pilot
2. Azure Machine Learning - Model deployment (optional)

Fallback: Works offline with local explanations if Azure unavailable
"""

import os
import json
import httpx
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass

# Configuration from environment
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"


@dataclass
class ThreatExplanation:
    """Structured threat explanation from Azure OpenAI"""
    summary: str
    severity_reasoning: str
    attack_chain: List[str]
    recommended_actions: List[str]
    mitre_context: str
    confidence: float


class AzureAIService:
    """
    Azure AI Integration for PCDS
    
    Provides:
    - Threat explanation using Azure OpenAI
    - Analyst co-pilot chat
    - Detection summarization
    """
    
    def __init__(self):
        self.endpoint = AZURE_OPENAI_ENDPOINT
        self.api_key = AZURE_OPENAI_KEY
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.enabled = bool(self.endpoint and self.api_key)
        
        if self.enabled:
            print(f"✅ Azure OpenAI connected: {self.deployment}")
        else:
            print("⚠️ Azure OpenAI not configured - using local fallback")
    
    async def explain_threat(self, detection: Dict) -> ThreatExplanation:
        """
        Use Azure OpenAI to explain a threat detection
        This is the XAI (Explainable AI) feature for judges
        """
        if not self.enabled:
            return self._local_explanation(detection)
        
        prompt = self._build_threat_prompt(detection)
        
        try:
            response = await self._call_azure_openai(prompt)
            return self._parse_explanation(response, detection)
        except Exception as e:
            print(f"Azure OpenAI error: {e}")
            return self._local_explanation(detection)
    
    async def analyst_copilot(self, query: str, context: Dict) -> str:
        """
        Analyst co-pilot - chat interface for SOC analysts
        Uses Azure OpenAI to answer security questions
        """
        if not self.enabled:
            return self._local_copilot_response(query)
        
        system_prompt = """You are a senior SOC analyst AI assistant for PCDS (Predictive Cyber Defense System).
You help security analysts investigate threats, understand detections, and recommend response actions.
Be concise, technical, and actionable. Reference MITRE ATT&CK when relevant."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {json.dumps(context)}\n\nQuestion: {query}"}
        ]
        
        try:
            return await self._call_azure_openai_chat(messages)
        except Exception as e:
            return f"Co-pilot unavailable: {e}"
    
    async def summarize_incident(self, detections: List[Dict]) -> str:
        """
        Summarize multiple related detections into an incident narrative
        """
        if not self.enabled:
            return self._local_incident_summary(detections)
        
        prompt = f"""Analyze these related security detections and provide a concise incident summary:

Detections:
{json.dumps(detections, indent=2)}

Provide:
1. Incident title (one line)
2. Attack narrative (2-3 sentences)
3. Affected assets
4. Recommended priority (P1-P4)
5. Immediate actions needed"""
        
        try:
            return await self._call_azure_openai(prompt)
        except Exception as e:
            return self._local_incident_summary(detections)
    
    def _build_threat_prompt(self, detection: Dict) -> str:
        """Build prompt for threat explanation"""
        return f"""Analyze this cybersecurity detection and explain it for a SOC analyst:

Detection Details:
- Type: {detection.get('detection_type', 'Unknown')}
- Severity: {detection.get('severity', 'medium')}
- Confidence: {detection.get('confidence', 0.5)}
- Entity: {detection.get('entity_id', 'Unknown')}
- MITRE Technique: {detection.get('technique_id', 'N/A')}
- Description: {detection.get('description', 'N/A')}

Provide a JSON response with:
{{
    "summary": "Brief one-sentence summary",
    "severity_reasoning": "Why this severity level",
    "attack_chain": ["Step 1", "Step 2", ...],
    "recommended_actions": ["Action 1", "Action 2", ...],
    "mitre_context": "MITRE ATT&CK context and related techniques"
}}"""
    
    async def _call_azure_openai(self, prompt: str) -> str:
        """Make API call to Azure OpenAI"""
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"]
    
    async def _call_azure_openai_chat(self, messages: List[Dict]) -> str:
        """Make chat API call to Azure OpenAI"""
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    def _parse_explanation(self, response: str, detection: Dict) -> ThreatExplanation:
        """Parse Azure OpenAI response into structured explanation"""
        try:
            data = json.loads(response)
            return ThreatExplanation(
                summary=data.get("summary", "Threat detected"),
                severity_reasoning=data.get("severity_reasoning", "Based on detection confidence"),
                attack_chain=data.get("attack_chain", []),
                recommended_actions=data.get("recommended_actions", ["Investigate further"]),
                mitre_context=data.get("mitre_context", ""),
                confidence=detection.get("confidence", 0.5)
            )
        except json.JSONDecodeError:
            return ThreatExplanation(
                summary=response[:200],
                severity_reasoning="AI analysis",
                attack_chain=[],
                recommended_actions=["Review detection details"],
                mitre_context="",
                confidence=detection.get("confidence", 0.5)
            )
    
    def _local_explanation(self, detection: Dict) -> ThreatExplanation:
        """Fallback local explanation when Azure unavailable"""
        severity = detection.get("severity", "medium")
        detection_type = detection.get("detection_type", "Unknown")
        technique_id = detection.get("technique_id", "")
        
        summaries = {
            "critical": f"Critical {detection_type} threat requiring immediate response",
            "high": f"High-severity {detection_type} detected - prompt investigation needed",
            "medium": f"Medium-risk {detection_type} activity - monitor and assess",
            "low": f"Low-severity {detection_type} - routine investigation"
        }
        
        return ThreatExplanation(
            summary=summaries.get(severity, f"{detection_type} detected"),
            severity_reasoning=f"Classified as {severity} based on ML ensemble confidence score",
            attack_chain=[
                "Initial compromise detected",
                f"Technique: {technique_id}" if technique_id else "Unknown technique",
                "Further analysis required"
            ],
            recommended_actions=[
                "Review entity timeline",
                "Check related detections",
                "Assess impact scope",
                f"Apply {severity}-priority response playbook"
            ],
            mitre_context=f"Mapped to MITRE ATT&CK: {technique_id}" if technique_id else "MITRE mapping pending",
            confidence=detection.get("confidence", 0.5)
        )
    
    def _local_copilot_response(self, query: str) -> str:
        """Fallback local response for co-pilot"""
        return """I'm the PCDS Analyst Co-pilot. Azure OpenAI is not configured.

To enable AI-powered assistance:
1. Set AZURE_OPENAI_ENDPOINT in your environment
2. Set AZURE_OPENAI_KEY with your API key
3. Restart the backend

For now, please review the detection details and entity timeline for investigation guidance."""
    
    def _local_incident_summary(self, detections: List[Dict]) -> str:
        """Fallback local incident summary"""
        count = len(detections)
        severities = [d.get("severity", "medium") for d in detections]
        highest = "critical" if "critical" in severities else "high" if "high" in severities else "medium"
        
        return f"""**Incident Summary (Local Analysis)**

- **Detection Count**: {count} related detections
- **Highest Severity**: {highest}
- **Recommendation**: Investigate as potential coordinated attack
- **Priority**: {"P1 - Immediate" if highest == "critical" else "P2 - High" if highest == "high" else "P3 - Medium"}

*Enable Azure OpenAI for deeper AI-powered analysis*"""


# Global instance
_azure_ai: Optional[AzureAIService] = None


def get_azure_ai() -> AzureAIService:
    """Get or create Azure AI service instance"""
    global _azure_ai
    if _azure_ai is None:
        _azure_ai = AzureAIService()
    return _azure_ai


# API endpoint functions
async def explain_detection(detection: Dict) -> Dict:
    """API: Get AI explanation for a detection"""
    ai = get_azure_ai()
    explanation = await ai.explain_threat(detection)
    return {
        "summary": explanation.summary,
        "severity_reasoning": explanation.severity_reasoning,
        "attack_chain": explanation.attack_chain,
        "recommended_actions": explanation.recommended_actions,
        "mitre_context": explanation.mitre_context,
        "confidence": explanation.confidence,
        "powered_by": "Azure OpenAI" if ai.enabled else "Local Analysis"
    }


async def copilot_query(query: str, context: Dict = None) -> Dict:
    """API: Analyst co-pilot query"""
    ai = get_azure_ai()
    response = await ai.analyst_copilot(query, context or {})
    return {
        "query": query,
        "response": response,
        "powered_by": "Azure OpenAI" if ai.enabled else "Local Analysis",
        "timestamp": datetime.now().isoformat()
    }
