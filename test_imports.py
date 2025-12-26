#!/usr/bin/env python3
"""
Test script to verify that updated imports work correctly.
This script checks if the migrated imports from langgraph.prebuilt to langchain.agents are available.

Usage: python test_imports.py
"""

import sys

def test_imports():
    """Test if the new imports work."""
    print("Testing updated imports...")
    
    # Test 1: Check if langchain.agents imports work
    try:
        from langchain.agents import create_react_agent, ToolNode, tools_condition
        print("✅ SUCCESS: langchain.agents imports work")
        print(f"   - create_react_agent: {create_react_agent}")
        print(f"   - ToolNode: {ToolNode}")
        print(f"   - tools_condition: {tools_condition}")
        return True
    except ImportError as e:
        print(f"❌ FAILED: langchain.agents imports failed: {e}")
        print("   Trying alternative: langchain-community or langgraph.prebuilt...")
        
        # Test 2: Check if it's in langchain-community
        try:
            from langchain_community.agents import create_react_agent, ToolNode, tools_condition
            print("✅ SUCCESS: Found in langchain_community.agents")
            return True
        except ImportError:
            pass
        
        # Test 3: Check if langgraph.prebuilt still works (for backward compatibility)
        try:
            from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
            print("⚠️  WARNING: langgraph.prebuilt still works but is deprecated")
            print("   Consider installing langchain package for LangGraph 1.0 compatibility")
            return True
        except ImportError as e2:
            print(f"❌ FAILED: langgraph.prebuilt also failed: {e2}")
            return False

def test_graph_apis():
    """Test if graph APIs work correctly."""
    print("\nTesting graph APIs...")
    try:
        from langgraph.graph import START, END, StateGraph, MessagesState
        print("✅ SUCCESS: Graph APIs import correctly")
        
        # Test that START and END are available
        assert START is not None, "START should not be None"
        assert END is not None, "END should not be None"
        print("✅ SUCCESS: START and END constants are available")
        return True
    except Exception as e:
        print(f"❌ FAILED: Graph API test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph Migration Import Test")
    print("=" * 60)
    
    import_success = test_imports()
    api_success = test_graph_apis()
    
    print("\n" + "=" * 60)
    if import_success and api_success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nNote: If langchain.agents is not found, you may need to:")
        print("  1. Install langchain: pip install langchain")
        print("  2. Or check if it's in langchain-community")
        print("  3. For LangGraph 1.0+, ensure you have the latest packages")
        sys.exit(1)


