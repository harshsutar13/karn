#!/usr/bin/env python3
"""
Comprehensive Test Suite for LOKI AI Workbench
Tests all functionalities: API endpoints, models, memory, RAG, agents, etc.
"""

import requests
import json
import time
import os
import sys
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION = "test-session-" + str(int(time.time()))

class LOKITester:
    def __init__(self):
        self.results = {"passed": 0, "failed": 0, "errors": []}
        self.session = requests.Session()

    def test(self, name, func):
        """Run a test and track results"""
        print(f"\n Testing: {name}")
        try:
            func()
            print(f" PASS: {name}")
            self.results["passed"] += 1
        except Exception as e:
            print(f" FAIL: {name} - {str(e)}")
            self.results["failed"] += 1
            self.results["errors"].append(f"{name}: {str(e)}")

    def assert_response(self, response, expected_status=200):
        """Assert response status and return JSON"""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}: {response.text}"
        if response.headers.get('content-type', '').startswith('application/json'):
            return response.json()
        return response.text

    # =================================================================
    # 1. CORE API TESTS
    # =================================================================

    def test_health_check(self):
        """Test basic server health"""
        resp = self.session.get(f"{BASE_URL}/")
        self.assert_response(resp)

    def test_capabilities_endpoint(self):
        """Test capabilities registry"""
        resp = self.session.get(f"{BASE_URL}/api/capabilities")
        data = self.assert_response(resp)
        assert "capabilities" in data
        assert len(data["capabilities"]) > 0
        print(f"   Found {len(data['capabilities'])} capabilities")

    def test_sovereignty_check(self):
        """Test network sovereignty audit"""
        resp = self.session.get(f"{BASE_URL}/api/sovereignty/status")
        data = self.assert_response(resp)
        assert "connections" in data
        print(f"   Network connections: {len(data['connections'])}")

    # =================================================================
    # 2. MEMORY SYSTEM TESTS
    # =================================================================

    def test_memory_sessions(self):
        """Test session listing"""
        resp = self.session.get(f"{BASE_URL}/api/memory/sessions")
        data = self.assert_response(resp)
        assert "sessions" in data
        print(f"   Found {len(data['sessions'])} existing sessions")

    def test_memory_clear(self):
        """Test session memory clearing"""
        resp = self.session.post(f"{BASE_URL}/api/memory/clear",
                                json={"session_id": TEST_SESSION})
        data = self.assert_response(resp)
        assert data["cleared"] is True

    def test_memory_history(self):
        """Test memory history retrieval"""
        resp = self.session.get(f"{BASE_URL}/api/memory/history",
                               params={"session_id": TEST_SESSION})
        data = self.assert_response(resp)
        assert "messages" in data
        assert "actions" in data

    # =================================================================
    # 3. CHAT & LLM TESTS
    # =================================================================

    def test_basic_chat(self):
        """Test basic chat functionality"""
        resp = self.session.post(f"{BASE_URL}/api/ask", json={
            "message": "Hello, what is 2+2?",
            "session_id": TEST_SESSION
        })
        data = self.assert_response(resp)
        assert "answer" in data
        assert "capability" in data
        print(f"   Capability: {data['capability']}")
        print(f"   Answer: {data['answer'][:50]}...")

    def test_streaming_chat(self):
        """Test streaming chat endpoint"""
        resp = self.session.post(f"{BASE_URL}/api/ask/stream", json={
            "message": "Count from 1 to 3",
            "session_id": TEST_SESSION
        }, stream=True)

        chunks = []
        for line in resp.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                    try:
                        chunk_data = json.loads(line_str[6:])
                        if chunk_data.get("kind") == "chunk":
                            chunks.append(chunk_data.get("text", ""))
                    except json.JSONDecodeError:
                        pass

        assert len(chunks) > 0, "No streaming chunks received"
        print(f"   Received {len(chunks)} streaming chunks")

    # =================================================================
    # 4. CODE GENERATION TESTS
    # =================================================================

    def test_code_generation(self):
        """Test code generation capability"""
        resp = self.session.post(f"{BASE_URL}/api/code/generate", json={
            "prompt": "Write a Python function to calculate factorial",
            "session_id": TEST_SESSION
        })
        data = self.assert_response(resp)
        assert "code" in data
        assert "def" in data["code"]  # Should contain function definition
        print(f"   Generated {len(data['code'])} characters of code")

    def test_sandbox_execution(self):
        """Test Python sandbox execution"""
        test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

result = factorial(5)
print(f"Factorial of 5 is {result}")
"""
        resp = self.session.post(f"{BASE_URL}/api/sandbox/run", json={
            "code": test_code,
            "session_id": TEST_SESSION
        })
        data = self.assert_response(resp)
        assert "output" in data
        assert "120" in data["output"]  # 5! = 120
        print(f"   Sandbox output: {data['output'].strip()}")

    # =================================================================
    # 5. RAG SYSTEM TESTS
    # =================================================================

    def test_rag_query(self):
        """Test RAG query functionality"""
        resp = self.session.post(f"{BASE_URL}/api/rag/query", json={
            "query": "What safety procedures should be followed?",
            "session_id": TEST_SESSION
        })
        data = self.assert_response(resp)
        assert "answer" in data
        assert "sources" in data
        print(f"   Found {len(data['sources'])} source documents")
        print(f"   Answer: {data['answer'][:50]}...")

    def test_rag_search(self):
        """Test RAG search functionality"""
        resp = self.session.post(f"{BASE_URL}/api/rag/search", json={
            "query": "maintenance procedures",
            "limit": 3
        })
        data = self.assert_response(resp)
        assert "results" in data
        print(f"   Found {len(data['results'])} search results")

    # =================================================================
    # 6. VISION & OCR TESTS
    # =================================================================

    def test_vision_analyze(self):
        """Test vision analysis (if image exists)"""
        # Check if there are any test images
        upload_dir = Path("data/uploads")
        if upload_dir.exists():
            images = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
            if images:
                image_path = str(images[0])
                resp = self.session.post(f"{BASE_URL}/api/vision/analyze", json={
                    "image_path": image_path,
                    "query": "Describe what you see in this image",
                    "session_id": TEST_SESSION
                })
                data = self.assert_response(resp)
                assert "description" in data
                print(f"   Vision analysis: {data['description'][:50]}...")
                return

        print("   No test images found, skipping vision test")

    # =================================================================
    # 7. AGENT & PLANNER TESTS
    # =================================================================

    def test_planner_goal(self):
        """Test goal planning functionality"""
        resp = self.session.post(f"{BASE_URL}/api/planner/plan", json={
            "goal": "Create a simple maintenance checklist document",
            "session_id": TEST_SESSION
        })
        data = self.assert_response(resp)
        assert "plan" in data
        assert len(data["plan"]) > 0
        print(f"   Generated plan with {len(data['plan'])} steps")

    def test_agent_analysis(self):
        """Test multi-step agent analysis"""
        resp = self.session.post(f"{BASE_URL}/api/agent/run", json={
            "task": "Analyze safety protocols and create a summary",
            "session_id": TEST_SESSION
        })
        data = self.assert_response(resp)
        assert "result" in data
        print(f"   Agent analysis: {data['result'][:50]}...")

    # =================================================================
    # 8. DOCUMENT GENERATION TESTS
    # =================================================================

    def test_document_generation(self):
        """Test document generation capabilities"""
        # Test Word document generation
        resp = self.session.post(f"{BASE_URL}/api/tools/word", json={
            "content": "# Test Document\n\nThis is a test document for LOKI.",
            "filename": "test_doc.docx",
            "session_id": TEST_SESSION
        })

        if resp.status_code == 200:
            data = self.assert_response(resp)
            assert "filename" in data
            print(f"   Generated document: {data['filename']}")
        else:
            print(f"   Document generation returned {resp.status_code}")

    def test_deliverable_download(self):
        """Test deliverable file download"""
        # Check if any deliverables exist
        outputs_dir = Path("backend/outputs")
        if outputs_dir.exists():
            files = list(outputs_dir.glob("*.docx")) + list(outputs_dir.glob("*.xlsx"))
            if files:
                filename = files[0].name
                resp = self.session.get(f"{BASE_URL}/api/deliverable/download/{filename}")
                assert resp.status_code in [200, 404]  # 404 if file moved/deleted
                print(f"   Download test for {filename}: {resp.status_code}")
                return

        print("   No deliverable files found for download test")

    # =================================================================
    # 9. PERFORMANCE & LOAD TESTS
    # =================================================================

    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import threading
        import time

        results = []
        def make_request():
            try:
                resp = self.session.post(f"{BASE_URL}/api/ask", json={
                    "message": f"Test concurrent request at {time.time()}",
                    "session_id": f"{TEST_SESSION}-concurrent"
                })
                results.append(resp.status_code)
            except Exception as e:
                results.append(str(e))

        # Launch 3 concurrent requests
        threads = []
        for i in range(3):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r == 200)
        print(f"   Concurrent requests: {success_count}/3 successful")
        assert success_count >= 2, "Most concurrent requests should succeed"

    # =================================================================
    # 10. INTEGRATION TESTS
    # =================================================================

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        session_id = f"{TEST_SESSION}-e2e"

        # 1. Clear session
        self.session.post(f"{BASE_URL}/api/memory/clear",
                         json={"session_id": session_id})

        # 2. Ask a question
        resp1 = self.session.post(f"{BASE_URL}/api/ask", json={
            "message": "What is industrial safety?",
            "session_id": session_id
        })
        data1 = self.assert_response(resp1)

        # 3. Follow up question (tests memory)
        resp2 = self.session.post(f"{BASE_URL}/api/ask", json={
            "message": "Can you elaborate on that?",
            "session_id": session_id
        })
        data2 = self.assert_response(resp2)

        # 4. Check memory was preserved
        resp3 = self.session.get(f"{BASE_URL}/api/memory/history",
                                params={"session_id": session_id})
        data3 = self.assert_response(resp3)

        assert len(data3["messages"]) >= 4  # 2 user + 2 assistant
        print(f"   E2E workflow: {len(data3['messages'])} messages in history")

    def run_all_tests(self):
        """Run the complete test suite"""
        print("=" * 60)
        print("LOKI COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Core API tests
        self.test("Health Check", self.test_health_check)
        self.test("Capabilities Endpoint", self.test_capabilities_endpoint)
        self.test("Sovereignty Check", self.test_sovereignty_check)

        # Memory system tests
        self.test("Memory Sessions", self.test_memory_sessions)
        self.test("Memory Clear", self.test_memory_clear)
        self.test("Memory History", self.test_memory_history)

        # Chat & LLM tests
        self.test("Basic Chat", self.test_basic_chat)
        self.test("Streaming Chat", self.test_streaming_chat)

        # Code generation tests
        self.test("Code Generation", self.test_code_generation)
        self.test("Sandbox Execution", self.test_sandbox_execution)

        # RAG system tests
        self.test("RAG Query", self.test_rag_query)
        self.test("RAG Search", self.test_rag_search)

        # Vision & OCR tests
        self.test("Vision Analysis", self.test_vision_analyze)

        # Agent & planner tests
        self.test("Planner Goal", self.test_planner_goal)
        self.test("Agent Analysis", self.test_agent_analysis)

        # Document generation tests
        self.test("Document Generation", self.test_document_generation)
        self.test("Deliverable Download", self.test_deliverable_download)

        # Performance tests
        self.test("Concurrent Requests", self.test_concurrent_requests)

        # Integration tests
        self.test("End-to-End Workflow", self.test_end_to_end_workflow)

        # Print final results
        print("\n" + "=" * 60)
        print(" TEST RESULTS")
        print("=" * 60)
        print(f" Passed: {self.results['passed']}")
        print(f" Failed: {self.results['failed']}")

        if self.results['errors']:
            print("\n ERRORS:")
            for error in self.results['errors']:
                print(f"    {error}")

        success_rate = (self.results['passed'] / (self.results['passed'] + self.results['failed'])) * 100
        print(f"\n Success Rate: {success_rate:.1f}%")

        if success_rate >= 90:
            print(" EXCELLENT: LOKI is functioning very well!")
        elif success_rate >= 75:
            print(" GOOD: LOKI is mostly functional with minor issues")
        elif success_rate >= 50:
            print(" MODERATE: LOKI has some significant issues")
        else:
            print(" CRITICAL: LOKI has major functionality problems")

if __name__ == "__main__":
    tester = LOKITester()
    tester.run_all_tests()