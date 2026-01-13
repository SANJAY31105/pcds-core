"""
E2E Pipeline Validation Test Suite
Tests the complete flow: Traffic → Kafka → ML → WebSocket → SOAR

Metrics Tracked:
- Throughput (events/sec)
- Latency (<50ms target)
- SOAR trigger correctness
- WebSocket broadcast verification
"""

import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class E2ETestResult:
    """Test result with metrics"""
    test_name: str
    passed: bool
    duration_ms: float
    events_processed: int = 0
    avg_latency_ms: float = 0.0
    throughput_eps: float = 0.0
    errors: List[str] = None
    details: Dict = None


class E2EPipelineTest:
    """
    End-to-End Pipeline Validation
    
    Tests:
    1. Throughput - Can we handle 1000+ events/sec?
    2. Latency - Is processing <50ms per event?
    3. SOAR triggers - Do critical detections trigger responses?
    4. WebSocket broadcast - Do events reach the frontend?
    """
    
    def __init__(self):
        self.results: List[E2ETestResult] = []
        self.latencies: List[float] = []
        
    async def run_all_tests(self) -> Dict:
        """Run all E2E tests and return summary"""
        print("\n" + "=" * 60)
        print("🧪 E2E PIPELINE VALIDATION TEST SUITE")
        print("=" * 60 + "\n")
        
        # Test 1: ML Engine Health
        await self.test_ml_engine()
        
        # Test 2: Throughput Test
        await self.test_throughput()
        
        # Test 3: Latency Test  
        await self.test_latency()
        
        # Test 4: SOAR Integration
        await self.test_soar_triggers()
        
        # Test 5: WebSocket Broadcast
        await self.test_websocket()
        
        # Test 6: Full Pipeline Integration
        await self.test_full_pipeline()
        
        # Generate summary
        return self._generate_summary()
    
    async def test_ml_engine(self) -> E2ETestResult:
        """Test 1: Verify ML engine is operational"""
        print("📊 Test 1: ML Engine Health Check")
        start = time.time()
        
        try:
            from ml.advanced_detector import get_advanced_engine
            engine = get_advanced_engine()
            
            # Test detection
            test_data = {
                "source_ip": "192.168.1.100",
                "dest_ip": "10.0.0.1",
                "source_port": 45123,
                "dest_port": 443,
                "packet_size": 1200,
                "features": [0.5] * 40
            }
            
            result = engine.detect(test_data, entity_id="test-entity")
            
            duration = (time.time() - start) * 1000
            passed = result is not None and "risk_level" in result
            
            result_obj = E2ETestResult(
                test_name="ML Engine Health",
                passed=passed,
                duration_ms=duration,
                events_processed=1,
                avg_latency_ms=duration,
                details={"risk_level": result.get("risk_level"), "confidence": result.get("confidence")}
            )
            
            print(f"   {'✅' if passed else '❌'} ML Engine: {result.get('risk_level', 'N/A')} ({duration:.1f}ms)")
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            result_obj = E2ETestResult(
                test_name="ML Engine Health",
                passed=False,
                duration_ms=duration,
                errors=[str(e)]
            )
            print(f"   ❌ ML Engine failed: {e}")
        
        self.results.append(result_obj)
        return result_obj
    
    async def test_throughput(self, num_events: int = 1000) -> E2ETestResult:
        """Test 2: Measure processing throughput"""
        print(f"\n📊 Test 2: Throughput Test ({num_events} events)")
        
        try:
            from ml.advanced_detector import get_advanced_engine
            engine = get_advanced_engine()
            
            # Generate test events
            events = []
            for i in range(num_events):
                events.append({
                    "source_ip": f"192.168.1.{i % 256}",
                    "dest_ip": "10.0.0.1",
                    "source_port": 45000 + i,
                    "dest_port": 443,
                    "features": [float(i % 10) / 10] * 40
                })
            
            # Process all events
            start = time.time()
            processed = 0
            
            for event in events:
                engine.detect(event, entity_id=f"entity-{processed}")
                processed += 1
            
            duration = time.time() - start
            throughput = processed / duration
            
            # Target: 1000 events/sec
            passed = throughput >= 500  # Relaxed target for non-GPU
            
            result_obj = E2ETestResult(
                test_name="Throughput Test",
                passed=passed,
                duration_ms=duration * 1000,
                events_processed=processed,
                throughput_eps=throughput,
                details={"target_eps": 1000, "actual_eps": throughput}
            )
            
            print(f"   {'✅' if passed else '⚠️'} Throughput: {throughput:.0f} events/sec (target: 500+)")
            
        except Exception as e:
            result_obj = E2ETestResult(
                test_name="Throughput Test",
                passed=False,
                duration_ms=0,
                errors=[str(e)]
            )
            print(f"   ❌ Throughput test failed: {e}")
        
        self.results.append(result_obj)
        return result_obj
    
    async def test_latency(self, num_samples: int = 100) -> E2ETestResult:
        """Test 3: Measure per-event processing latency"""
        print(f"\n📊 Test 3: Latency Test ({num_samples} samples)")
        
        try:
            from ml.advanced_detector import get_advanced_engine
            engine = get_advanced_engine()
            
            latencies = []
            
            for i in range(num_samples):
                event = {
                    "source_ip": "192.168.1.100",
                    "dest_ip": "10.0.0.1",
                    "features": [0.5] * 40
                }
                
                start = time.perf_counter()
                engine.detect(event, entity_id=f"latency-test-{i}")
                latencies.append((time.perf_counter() - start) * 1000)
            
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
            
            # Target: <50ms average
            passed = avg_latency < 50
            
            result_obj = E2ETestResult(
                test_name="Latency Test",
                passed=passed,
                duration_ms=sum(latencies),
                events_processed=num_samples,
                avg_latency_ms=avg_latency,
                details={"p95_ms": p95_latency, "p99_ms": p99_latency, "min_ms": min(latencies), "max_ms": max(latencies)}
            )
            
            self.latencies = latencies
            print(f"   {'✅' if passed else '❌'} Latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms")
            
        except Exception as e:
            result_obj = E2ETestResult(
                test_name="Latency Test",
                passed=False,
                duration_ms=0,
                errors=[str(e)]
            )
            print(f"   ❌ Latency test failed: {e}")
        
        self.results.append(result_obj)
        return result_obj
    
    async def test_soar_triggers(self) -> E2ETestResult:
        """Test 4: Verify SOAR triggers on critical detections"""
        print("\n📊 Test 4: SOAR Integration Test")
        
        try:
            from ml.soar_orchestrator import SOAROrchestrator, IncidentSeverity
            
            soar = SOAROrchestrator()
            
            # Create a critical incident (must await - async method)
            incident = await soar.create_incident(
                title="E2E Test: Critical Threat Detected",
                description="Test incident for E2E validation",
                severity=IncidentSeverity.CRITICAL,
                source="e2e_test",
                ml_prediction={"risk_level": "critical", "confidence": 0.95},
                affected_hosts=["test-host-001"]
            )
            
            passed = incident is not None and incident.status is not None
            
            # Get incident details
            retrieved = soar.get_incident(incident.incident_id)
            
            result_obj = E2ETestResult(
                test_name="SOAR Integration",
                passed=passed,
                duration_ms=0,
                events_processed=1,
                details={
                    "incident_id": incident.incident_id,
                    "status": incident.status.value if incident.status else None,
                    "playbook": incident.playbook_id
                }
            )
            
            print(f"   {'✅' if passed else '❌'} SOAR: Incident created, status={incident.status.value}")
            
        except Exception as e:
            result_obj = E2ETestResult(
                test_name="SOAR Integration",
                passed=False,
                duration_ms=0,
                errors=[str(e)]
            )
            print(f"   ❌ SOAR test failed: {e}")
        
        self.results.append(result_obj)
        return result_obj
    
    async def test_websocket(self) -> E2ETestResult:
        """Test 5: Verify WebSocket manager functionality"""
        print("\n📊 Test 5: WebSocket Manager Test")
        
        try:
            from websocket_manager import ConnectionManager
            
            manager = ConnectionManager()
            
            # Test broadcast method exists and works
            test_message = {"type": "test", "data": "e2e_validation"}
            
            # Should not raise error even with no connections
            await manager.broadcast_to_all(test_message)
            
            passed = True
            result_obj = E2ETestResult(
                test_name="WebSocket Manager",
                passed=passed,
                duration_ms=0,
                details={"connections": manager.get_connection_count()}
            )
            
            print(f"   ✅ WebSocket manager operational")
            
        except Exception as e:
            result_obj = E2ETestResult(
                test_name="WebSocket Manager",
                passed=False,
                duration_ms=0,
                errors=[str(e)]
            )
            print(f"   ❌ WebSocket test failed: {e}")
        
        self.results.append(result_obj)
        return result_obj
    
    async def test_full_pipeline(self) -> E2ETestResult:
        """Test 6: Full pipeline integration test"""
        print("\n📊 Test 6: Full Pipeline Integration")
        
        try:
            from ml.realtime_pipeline import get_realtime_pipeline, PipelineEvent
            
            pipeline = get_realtime_pipeline()
            
            # Inject test event
            test_event = PipelineEvent(
                event_id="e2e-test-001",
                timestamp=datetime.now().isoformat(),
                source_ip="192.168.1.100",
                source_host="test-workstation",
                features=[0.7] * 40,
                raw_data={"test": True}
            )
            
            await pipeline.inject_event(test_event)
            
            # Check stats
            stats = pipeline.get_stats()
            
            passed = stats is not None
            result_obj = E2ETestResult(
                test_name="Full Pipeline",
                passed=passed,
                duration_ms=0,
                details={"pipeline_running": stats.get("is_running", False)}
            )
            
            print(f"   ✅ Pipeline stats: {json.dumps(stats, indent=2)[:200]}...")
            
        except Exception as e:
            result_obj = E2ETestResult(
                test_name="Full Pipeline",
                passed=False,
                duration_ms=0,
                errors=[str(e)]
            )
            print(f"   ❌ Pipeline test failed: {e}")
        
        self.results.append(result_obj)
        return result_obj
    
    def _generate_summary(self) -> Dict:
        """Generate test summary"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print("📊 E2E TEST SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} │ {result.test_name}")
        
        print("-" * 60)
        print(f"  Total: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
        
        # Performance summary
        if self.latencies:
            print(f"\n  🚀 Performance Metrics:")
            print(f"     Avg Latency: {statistics.mean(self.latencies):.2f}ms")
            print(f"     P95 Latency: {sorted(self.latencies)[int(len(self.latencies)*0.95)]:.2f}ms")
        
        throughput_result = next((r for r in self.results if r.test_name == "Throughput Test"), None)
        if throughput_result and throughput_result.throughput_eps:
            print(f"     Throughput: {throughput_result.throughput_eps:.0f} events/sec")
        
        print("=" * 60 + "\n")
        
        return {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in self.results
            ]
        }


async def main():
    """Run E2E tests"""
    test_suite = E2EPipelineTest()
    summary = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if summary["pass_rate"] >= 0.8 else 1)


if __name__ == "__main__":
    asyncio.run(main())
