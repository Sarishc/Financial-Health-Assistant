"""
Performance tests for application startup and loading
"""
import os
import sys
import time
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import importlib
import psutil
import requests
import multiprocessing
import subprocess
import signal
import socket
from contextlib import closing

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def find_free_port():
    """Find a free port to use for testing"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

class AppProcess:
    """Context manager to start and stop the application server"""
    
    def __init__(self, port=None):
        self.port = port if port is not None else find_free_port()
        self.process = None
    
    def __enter__(self):
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(self.port)]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for the server to start
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        else:
            self.process.terminate()
            raise TimeoutError("Server did not start within the timeout period")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            self.process.terminate()
            self.process.wait()

def test_module_import_time():
    """Test the import time for key modules"""
    print("\n===== Module Import Performance Test =====")
    
    modules_to_test = [
        'app.data.processor',
        'app.models.categorization.nlp_categorizer',
        'app.models.time_series.time_series_processor',
        'app.models.forecasting.spending_forecaster',
        'app.models.recommendation.recommendation_engine',
        'app.utils.performance',
        'app.utils.caching'
    ]
    
    results = []
    
    for module_name in modules_to_test:
        # Clear the module from sys.modules if it's already imported
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Measure import time
        start_time = time.time()
        module = importlib.import_module(module_name)
        import_time = time.time() - start_time
        
        results.append({
            'module': module_name,
            'time': import_time
        })
        
        print(f"Import time for {module_name}: {import_time:.4f} seconds")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.barh([r['module'].split('.')[-1] for r in results], [r['time'] for r in results])
    plt.xlabel('Time (seconds)')
    plt.title('Module Import Time')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig('module_import_time.png')
    
    # Check if any module takes too long to import
    for result in results:
        assert result['time'] < 2.0, f"Module {result['module']} takes too long to import"

def test_app_startup_time():
    """Test the application startup time"""
    print("\n===== Application Startup Performance Test =====")
    
    # Find a free port
    port = find_free_port()
    
    # Start and stop the application multiple times to get an average
    startup_times = []
    memory_usages = []
    num_runs = 3
    
    for i in range(num_runs):
        # Clear any cached modules
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('app.'):
                del sys.modules[module_name]
        
        # Start time measurement
        start_time = time.time()
        
        # Start the application in a separate process
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for the server to start
        while True:
            try:
                response = requests.get(f"http://localhost:{port}")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                # Check if the process is still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"Server process terminated: {stdout.decode()}, {stderr.decode()}")
                    pytest.fail("Server process terminated unexpectedly")
                time.sleep(0.1)
        
        # Calculate startup time
        startup_time = time.time() - start_time
        startup_times.append(startup_time)
        
        # Get memory usage
        process_obj = psutil.Process(process.pid)
        memory_usage = process_obj.memory_info().rss / (1024 * 1024)  # MB
        memory_usages.append(memory_usage)
        
        print(f"Run {i+1}: Startup time: {startup_time:.2f} seconds, Memory usage: {memory_usage:.2f} MB")
        
        # Stop the server
        process.terminate()
        process.wait()
    
    # Calculate averages
    avg_startup_time = sum(startup_times) / len(startup_times)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)
    
    print(f"Average startup time: {avg_startup_time:.2f} seconds")
    print(f"Average memory usage: {avg_memory_usage:.2f} MB")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, num_runs + 1), startup_times)
    plt.axhline(y=avg_startup_time, color='r', linestyle='--', label=f'Avg: {avg_startup_time:.2f}s')
    plt.xlabel('Run')
    plt.ylabel('Time (seconds)')
    plt.title('Application Startup Time')
    plt.grid(axis='y')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, num_runs + 1), memory_usages)
    plt.axhline(y=avg_memory_usage, color='r', linestyle='--', label=f'Avg: {avg_memory_usage:.2f} MB')
    plt.xlabel('Run')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Application Memory Usage')
    plt.grid(axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('app_startup_performance.png')
    
    # Assert performance requirements
    assert avg_startup_time < 5.0, "Application startup time is too slow"
    assert avg_memory_usage < 500.0, "Application memory usage is too high"

def test_api_endpoint_performance():
    """Test the performance of key API endpoints"""
    print("\n===== API Endpoint Performance Test =====")
    
    # Find a free port
    port = find_free_port()
    
    with AppProcess(port=port) as app:
        # Define endpoints to test
        endpoints = [
            '/',  # Health check
            '/api/v1/transactions',
            '/api/v1/categories',
            '/api/v1/forecasts',
            '/api/v1/recommendations'
        ]
        
        results = []
        
        for endpoint in endpoints:
            url = f"http://localhost:{port}{endpoint}"
            
            # Test response time (average of multiple requests)
            num_requests = 10
            response_times = []
            
            for _ in range(num_requests):
                try:
                    start_time = time.time()
                    response = requests.get(url)
                    response_time = time.time() - start_time
                    
                    response_times.append(response_time)
                    
                    # Sleep briefly to avoid overwhelming the server
                    time.sleep(0.1)
                except requests.exceptions.RequestException as e:
                    print(f"Error accessing {url}: {str(e)}")
                    break
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                results.append({
                    'endpoint': endpoint,
                    'avg_response_time': avg_response_time,
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times)
                })
                
                print(f"Endpoint {endpoint}: Avg: {avg_response_time:.4f}s, Min: {min(response_times):.4f}s, Max: {max(response_times):.4f}s")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        endpoints = [r['endpoint'] for r in results]
        avg_times = [r['avg_response_time'] for r in results]
        min_times = [r['min_response_time'] for r in results]
        max_times = [r['max_response_time'] for r in results]
        
        x = range(len(endpoints))
        plt.bar(x, avg_times, label='Average')
        plt.plot(x, min_times, 'go-', label='Minimum')
        plt.plot(x, max_times, 'ro-', label='Maximum')
        
        plt.xlabel('Endpoint')
        plt.ylabel('Response Time (seconds)')
        plt.title('API Endpoint Performance')
        plt.xticks(x, [e.split('/')[-1] or 'root' for e in endpoints], rotation=45)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('api_endpoint_performance.png')
        
        # Assert performance requirements
        for result in results:
            assert result['avg_response_time'] < 1.0, f"Endpoint {result['endpoint']} is too slow"

if __name__ == "__main__":
    # Run the tests
    test_module_import_time()
    test_app_startup_time()
    test_api_endpoint_performance()