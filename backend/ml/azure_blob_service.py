"""
Azure Blob Storage Service - PCDS Enterprise
Stores detection reports, ML models, and audit logs in Azure Blob Storage

This is the SECOND Azure service (alongside Azure OpenAI) for Imagine Cup compliance.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import io

# Azure SDK
try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    HAS_AZURE_BLOB = True
except ImportError:
    HAS_AZURE_BLOB = False
    print("⚠️ Azure Blob Storage SDK not installed. Run: pip install azure-storage-blob")


class AzureBlobService:
    """
    Azure Blob Storage integration for PCDS
    
    Containers:
    - pcds-reports: Detection and incident reports (PDF/JSON)
    - pcds-models: ML model files
    - pcds-logs: Audit logs
    """
    
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT", "pcdstorage")
        
        self.client: Optional[BlobServiceClient] = None
        self.containers = {
            "reports": "pcds-reports",
            "models": "pcds-models",
            "logs": "pcds-logs"
        }
        
        self._initialized = False
        
        if HAS_AZURE_BLOB and self.connection_string:
            self._initialize()
        else:
            print("📦 Azure Blob Storage: Running in demo mode (no connection string)")
    
    def _initialize(self):
        """Initialize Azure Blob Storage client"""
        try:
            self.client = BlobServiceClient.from_connection_string(self.connection_string)
            
            # Create containers if they don't exist
            for container_name in self.containers.values():
                try:
                    self.client.create_container(container_name)
                    print(f"  ✅ Created container: {container_name}")
                except ResourceExistsError:
                    pass  # Container already exists
            
            self._initialized = True
            print(f"✅ Azure Blob Storage initialized: {self.account_name}")
            
        except Exception as e:
            print(f"⚠️ Azure Blob Storage init failed: {e}")
            self._initialized = False
    
    @property
    def is_connected(self) -> bool:
        return self._initialized and self.client is not None
    
    # ==================== REPORT STORAGE ====================
    
    def upload_report(self, report_name: str, content: str, report_type: str = "json") -> Dict[str, Any]:
        """
        Upload a detection/incident report to Azure Blob Storage
        
        Args:
            report_name: Name of the report (e.g., "incident_2024_001")
            content: Report content (JSON string or text)
            report_type: "json" or "pdf"
        
        Returns:
            Upload result with blob URL
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"{timestamp}_{report_name}.{report_type}"
        
        if not self.is_connected:
            # Demo mode - simulate upload
            return {
                "success": True,
                "demo_mode": True,
                "blob_name": blob_name,
                "url": f"https://{self.account_name}.blob.core.windows.net/{self.containers['reports']}/{blob_name}",
                "size_bytes": len(content),
                "uploaded_at": datetime.now().isoformat()
            }
        
        try:
            container_client = self.client.get_container_client(self.containers["reports"])
            blob_client = container_client.get_blob_client(blob_name)
            
            # Upload content
            blob_client.upload_blob(content, overwrite=True)
            
            return {
                "success": True,
                "blob_name": blob_name,
                "url": blob_client.url,
                "size_bytes": len(content),
                "uploaded_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_detection_report(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload a detection event report"""
        report_name = f"detection_{detection_data.get('id', 'unknown')}"
        content = json.dumps(detection_data, indent=2, default=str)
        return self.upload_report(report_name, content, "json")
    
    def upload_incident_report(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload an incident report"""
        report_name = f"incident_{incident_data.get('id', 'unknown')}"
        content = json.dumps(incident_data, indent=2, default=str)
        return self.upload_report(report_name, content, "json")
    
    def list_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent reports from blob storage"""
        if not self.is_connected:
            # Demo mode - return sample data
            return [
                {
                    "name": "20241225_incident_ransomware_001.json",
                    "size": 2048,
                    "last_modified": datetime.now().isoformat(),
                    "url": f"https://{self.account_name}.blob.core.windows.net/{self.containers['reports']}/sample.json"
                }
            ]
        
        try:
            container_client = self.client.get_container_client(self.containers["reports"])
            blobs = container_client.list_blobs()
            
            reports = []
            for blob in blobs:
                reports.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    "url": f"https://{self.account_name}.blob.core.windows.net/{self.containers['reports']}/{blob.name}"
                })
                if len(reports) >= limit:
                    break
            
            return reports
            
        except Exception as e:
            return [{"error": str(e)}]
    
    def download_report(self, blob_name: str) -> Optional[str]:
        """Download a report from blob storage"""
        if not self.is_connected:
            return None
        
        try:
            container_client = self.client.get_container_client(self.containers["reports"])
            blob_client = container_client.get_blob_client(blob_name)
            
            download_stream = blob_client.download_blob()
            return download_stream.readall().decode('utf-8')
            
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            return None
    
    # ==================== AUDIT LOGGING ====================
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an audit event to Azure Blob Storage
        
        Args:
            event_type: Type of event (e.g., "detection", "response", "user_action")
            details: Event details
        """
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        # Create daily log file
        date_str = timestamp.strftime("%Y%m%d")
        blob_name = f"audit_log_{date_str}.jsonl"
        
        if not self.is_connected:
            return {
                "success": True,
                "demo_mode": True,
                "log_file": blob_name,
                "event_type": event_type
            }
        
        try:
            container_client = self.client.get_container_client(self.containers["logs"])
            blob_client = container_client.get_blob_client(blob_name)
            
            # Append to existing log or create new
            try:
                existing_data = blob_client.download_blob().readall().decode('utf-8')
            except ResourceNotFoundError:
                existing_data = ""
            
            new_data = existing_data + json.dumps(log_entry) + "\n"
            blob_client.upload_blob(new_data, overwrite=True)
            
            return {
                "success": True,
                "log_file": blob_name,
                "event_type": event_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== MODEL STORAGE ====================
    
    def upload_model(self, model_name: str, model_bytes: bytes) -> Dict[str, Any]:
        """Upload an ML model file to blob storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"{model_name}_{timestamp}.pkl"
        
        if not self.is_connected:
            return {
                "success": True,
                "demo_mode": True,
                "blob_name": blob_name,
                "size_bytes": len(model_bytes)
            }
        
        try:
            container_client = self.client.get_container_client(self.containers["models"])
            blob_client = container_client.get_blob_client(blob_name)
            
            blob_client.upload_blob(model_bytes, overwrite=True)
            
            return {
                "success": True,
                "blob_name": blob_name,
                "url": blob_client.url,
                "size_bytes": len(model_bytes)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # ==================== STATS ====================
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "service": "Azure Blob Storage",
            "account": self.account_name,
            "connected": self.is_connected,
            "containers": list(self.containers.values()),
            "demo_mode": not self.is_connected
        }
        
        if self.is_connected:
            try:
                # Count blobs in each container
                for key, container_name in self.containers.items():
                    container_client = self.client.get_container_client(container_name)
                    blobs = list(container_client.list_blobs())
                    stats[f"{key}_count"] = len(blobs)
            except:
                pass
        
        return stats


# Global instance
_blob_service: Optional[AzureBlobService] = None


def get_blob_service() -> AzureBlobService:
    """Get or create Azure Blob Storage service"""
    global _blob_service
    if _blob_service is None:
        _blob_service = AzureBlobService()
    return _blob_service


# Test function
if __name__ == "__main__":
    print("🧪 Testing Azure Blob Storage Service...")
    
    service = get_blob_service()
    print(f"\nConnection status: {service.is_connected}")
    print(f"Storage stats: {service.get_storage_stats()}")
    
    # Test upload (demo mode)
    result = service.upload_detection_report({
        "id": "test_001",
        "type": "ransomware",
        "severity": "critical",
        "timestamp": datetime.now().isoformat()
    })
    print(f"\nUpload result: {result}")
    
    # Test audit log
    log_result = service.log_audit_event("test", {"action": "unit_test"})
    print(f"Audit log result: {log_result}")
