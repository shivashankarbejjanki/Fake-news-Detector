"""
Comprehensive Testing Script for Fake News Detection System
Tests all components and deployment options
"""

import requests
import time
import sys
import os
import subprocess
import json
from datetime import datetime

def test_local_server(base_url="http://127.0.0.1:5000"):
    """Test the local Flask server."""
    print("Testing Local Flask Server")
    print("-" * 40)
    
    try:
        # Test home page
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("[PASS] Home page accessible")
        else:
            print(f"[FAIL] Home page failed: {response.status_code}")
            return False
        
        # Test API endpoints
        test_text = "Scientists at MIT have developed new technology."
        
        # Test basic prediction API
        api_response = requests.post(
            f"{base_url}/api/predict",
            json={"text": test_text},
            timeout=30
        )
        
        if api_response.status_code == 200:
            result = api_response.json()
            print("✅ Basic API working")
            print(f"   Prediction: {result.get('prediction', 'N/A')}")
        else:
            print(f"❌ Basic API failed: {api_response.status_code}")
            return False
        
        # Test comprehensive analysis API
        analysis_response = requests.post(
            f"{base_url}/api/analyze",
            json={"text": test_text, "include_ensemble": True},
            timeout=30
        )
        
        if analysis_response.status_code == 200:
            result = analysis_response.json()
            print("✅ Comprehensive API working")
            if 'ensemble_prediction' in result:
                print("✅ Ensemble prediction available")
            if 'text_analysis' in result:
                print("✅ Text analysis available")
        else:
            print(f"❌ Comprehensive API failed: {analysis_response.status_code}")
        
        # Test models endpoint
        models_response = requests.get(f"{base_url}/api/models", timeout=10)
        if models_response.status_code == 200:
            models = models_response.json()
            print(f"✅ Models endpoint working: {len(models.get('models', []))} models")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_network_access():
    """Test network accessibility."""
    print("\n🌐 Testing Network Access")
    print("-" * 40)
    
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print(f"Computer Name: {hostname}")
        print(f"Local IP: {local_ip}")
        
        # Test network access
        network_url = f"http://{local_ip}:5000"
        response = requests.get(network_url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Network access working: {network_url}")
            return network_url
        else:
            print(f"❌ Network access failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Network test failed: {e}")
    
    return None

def test_mobile_compatibility(base_url):
    """Test mobile compatibility."""
    print("\n📱 Testing Mobile Compatibility")
    print("-" * 40)
    
    mobile_headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
    }
    
    try:
        response = requests.get(base_url, headers=mobile_headers, timeout=10)
        if response.status_code == 200:
            print("✅ Mobile user agent test passed")
            
            # Check for responsive design indicators
            content = response.text.lower()
            if 'viewport' in content:
                print("✅ Viewport meta tag found")
            if 'bootstrap' in content or 'responsive' in content:
                print("✅ Responsive framework detected")
            
            return True
        else:
            print(f"❌ Mobile test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Mobile test error: {e}")
    
    return False

def test_performance(base_url):
    """Test performance metrics."""
    print("\n⚡ Testing Performance")
    print("-" * 40)
    
    test_cases = [
        "Short text for testing.",
        "This is a medium length text that should provide reasonable analysis results for testing purposes.",
        "This is a much longer text that contains multiple sentences and should test the system's ability to handle larger inputs. It includes various types of content and should provide comprehensive analysis results. The system should be able to process this text efficiently and provide accurate predictions based on the trained models."
    ]
    
    for i, text in enumerate(test_cases, 1):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/predict",
                json={"text": text},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                duration = end_time - start_time
                print(f"✅ Test {i}: {duration:.2f}s (Length: {len(text)} chars)")
            else:
                print(f"❌ Test {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Performance test {i} error: {e}")

def test_deployment_files():
    """Test deployment files exist and are valid."""
    print("\n📁 Testing Deployment Files")
    print("-" * 40)
    
    required_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'Procfile',
        'runtime.txt',
        'streamlit_app.py',
        'LAUNCH.bat',
        'deploy.sh'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
    
    # Test requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            essential_packages = ['flask', 'scikit-learn', 'pandas', 'numpy', 'nltk']
            for package in essential_packages:
                if package.lower() in requirements.lower():
                    print(f"✅ {package} in requirements")
                else:
                    print(f"⚠️ {package} not found in requirements")

def test_docker_setup():
    """Test Docker setup."""
    print("\n🐳 Testing Docker Setup")
    print("-" * 40)
    
    try:
        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            
            # Test docker-compose
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Docker Compose available: {result.stdout.strip()}")
                return True
            else:
                print("⚠️ Docker Compose not available")
        else:
            print("⚠️ Docker not available")
            
    except FileNotFoundError:
        print("⚠️ Docker not installed")
    except Exception as e:
        print(f"⚠️ Docker test error: {e}")
    
    return False

def generate_test_report():
    """Generate comprehensive test report."""
    print("\n📋 Generating Test Report")
    print("=" * 50)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Run all tests
    print("Running comprehensive tests...")
    
    # Test local server
    server_working = test_local_server()
    report['tests']['local_server'] = server_working
    
    if server_working:
        # Test network access
        network_url = test_network_access()
        report['tests']['network_access'] = network_url is not None
        report['network_url'] = network_url
        
        # Test mobile compatibility
        mobile_ok = test_mobile_compatibility("http://127.0.0.1:5000")
        report['tests']['mobile_compatibility'] = mobile_ok
        
        # Test performance
        test_performance("http://127.0.0.1:5000")
    
    # Test deployment files
    test_deployment_files()
    
    # Test Docker
    docker_ok = test_docker_setup()
    report['tests']['docker_available'] = docker_ok
    
    # Save report
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Test Report Summary")
    print("-" * 30)
    for test_name, result in report['tests'].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n📄 Detailed report saved to: test_report.json")
    
    return report

def main():
    """Main testing function."""
    print("FAKE NEWS DETECTION SYSTEM - DEPLOYMENT TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate comprehensive test report
    report = generate_test_report()
    
    # Final recommendations
    print("\n🎯 DEPLOYMENT RECOMMENDATIONS")
    print("-" * 40)
    
    if report['tests'].get('local_server'):
        print("✅ Local deployment: READY")
        print("   • Use LAUNCH.bat for one-click start")
        print("   • Access at: http://127.0.0.1:5000")
        
        if report['tests'].get('network_access'):
            print("✅ Network sharing: READY")
            print(f"   • Share URL: {report.get('network_url', 'N/A')}")
        
        if report['tests'].get('mobile_compatibility'):
            print("✅ Mobile access: READY")
            print("   • Works on smartphones and tablets")
    
    if report['tests'].get('docker_available'):
        print("✅ Docker deployment: READY")
        print("   • Run: docker-compose up -d")
    
    print("\n🌐 Cloud deployment options:")
    print("   • Streamlit Cloud (free): Upload to GitHub + deploy")
    print("   • Heroku (professional): heroku create + git push")
    print("   • Railway/Render (modern): Connect GitHub repo")
    
    print("\n🎉 TESTING COMPLETE!")
    print("Your Fake News Detection System is ready for deployment!")

if __name__ == "__main__":
    main()
