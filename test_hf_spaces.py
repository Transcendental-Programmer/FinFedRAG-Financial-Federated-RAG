#!/usr/bin/env python3
"""
Test script for Hugging Face Spaces deployment
Verifies that the app.py works correctly without external dependencies
"""

import sys
import os
import importlib.util

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    required_packages = [
        'streamlit',
        'numpy', 
        'pandas',
        'time',
        'threading',
        'json',
        'logging',
        'datetime',
        'random'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_app_structure():
    """Test that app.py has the required structure"""
    print("\nTesting app.py structure...")
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found")
        return False
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    required_components = [
        'SimulatedFederatedSystem',
        'ClientSimulator', 
        'st.set_page_config',
        'st.title',
        'st.sidebar',
        'st.header',
        'st.form'
    ]
    
    for component in required_components:
        if component in content:
            print(f"✅ {component}")
        else:
            print(f"❌ {component} not found")
            return False
    
    return True

def test_requirements():
    """Test that requirements.txt is minimal"""
    print("\nTesting requirements.txt...")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    # Check for minimal dependencies
    minimal_deps = ['streamlit', 'numpy', 'pandas']
    heavy_deps = ['tensorflow', 'torch', 'scikit-learn', 'flask', 'fastapi']
    
    for dep in minimal_deps:
        if dep in requirements:
            print(f"✅ {dep}")
        else:
            print(f"❌ {dep} missing")
            return False
    
    for dep in heavy_deps:
        if dep in requirements:
            print(f"⚠️  {dep} found (may cause HF Spaces issues)")
    
    return True

def test_readme():
    """Test that README.md has HF Spaces config"""
    print("\nTesting README.md...")
    
    if not os.path.exists('README.md'):
        print("❌ README.md not found")
        return False
    
    with open('README.md', 'r') as f:
        content = f.read()
    
    required_config = [
        'title: Federated Credit Scoring',
        'sdk: streamlit',
        'app_port: 8501'
    ]
    
    for config in required_config:
        if config in content:
            print(f"✅ {config}")
        else:
            print(f"❌ {config} not found")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Hugging Face Spaces Deployment")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_structure,
        test_requirements,
        test_readme
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error: {test.__name__} - {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for HF Spaces deployment.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 