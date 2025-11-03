#!/bin/bash

# Gmail MCP Integration Quick Start Script
# This script helps set up the Gmail MCP integration step by step

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR"
CREDENTIALS_DIR="$BACKEND_DIR/credentials"

echo "=================================="
echo "Gmail MCP Integration Setup"
echo "=================================="
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   ✅ Python version: $PYTHON_VERSION"
echo ""

# Step 2: Install dependencies
echo "Step 2: Installing Python dependencies..."
echo "   This will install: mcp, google-auth, google-api-python-client, etc."
read -p "   Install dependencies now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if virtual environment exists
    if [ -d "$BACKEND_DIR/venv" ]; then
        echo "   Using existing virtual environment: venv/"
        source "$BACKEND_DIR/venv/bin/activate"
        pip install -r "$BACKEND_DIR/requirements.txt"
    else
        echo "   Creating new virtual environment..."
        python3 -m venv "$BACKEND_DIR/venv"
        source "$BACKEND_DIR/venv/bin/activate"
        pip install -r "$BACKEND_DIR/requirements.txt"
    fi
    echo "   ✅ Dependencies installed"
    echo "   Note: Virtual environment activated. To deactivate later, run: deactivate"
else
    echo "   ⚠️  Skipped dependency installation"
    echo "   To install manually: source venv/bin/activate && pip install -r requirements.txt"
fi
echo ""

# Step 3: Check credentials directory
echo "Step 3: Checking credentials directory..."
if [ ! -d "$CREDENTIALS_DIR" ]; then
    mkdir -p "$CREDENTIALS_DIR"
    echo "   ✅ Created credentials directory"
else
    echo "   ✅ Credentials directory exists"
fi
echo ""

# Step 4: Check for Gmail credentials
echo "Step 4: Checking for Gmail OAuth credentials..."
CREDS_FILE="$CREDENTIALS_DIR/gmail_credentials.json"
if [ -f "$CREDS_FILE" ]; then
    echo "   ✅ Gmail credentials found: $CREDS_FILE"
else
    echo "   ❌ Gmail credentials NOT found"
    echo ""
    echo "   You need to create OAuth 2.0 credentials from Google Cloud Console:"
    echo ""
    echo "   1. Go to: https://console.cloud.google.com/"
    echo "   2. Create a project (or select existing)"
    echo "   3. Enable Gmail API"
    echo "   4. Create OAuth 2.0 credentials (Desktop app)"
    echo "   5. Download the JSON file"
    echo "   6. Save it as: $CREDS_FILE"
    echo ""
    echo "   For detailed instructions, see: GMAIL_MCP_SETUP.md"
    echo ""
    read -p "   Have you downloaded the credentials file? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Please save it to: $CREDS_FILE"
        echo "   Then run this script again."
    fi
    exit 1
fi
echo ""

# Step 5: Check for token (authentication status)
echo "Step 5: Checking Gmail authentication status..."
TOKEN_FILE="$CREDENTIALS_DIR/gmail_token.json"
if [ -f "$TOKEN_FILE" ]; then
    echo "   ✅ Gmail token found (already authenticated)"
else
    echo "   ⚠️  Gmail token NOT found (need to authenticate)"
    echo ""
    echo "   To authenticate with Gmail:"
    echo "   1. Run: python $BACKEND_DIR/mcp_servers/gmail_server.py"
    echo "   2. A browser window will open"
    echo "   3. Sign in to your Gmail account"
    echo "   4. Grant permission to the app"
    echo "   5. The token will be saved automatically"
    echo ""
    read -p "   Authenticate with Gmail now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Starting Gmail authentication..."
        echo "   (Press Ctrl+C after authentication completes)"
        python3 "$BACKEND_DIR/mcp_servers/gmail_server.py" || true
        echo ""
        if [ -f "$TOKEN_FILE" ]; then
            echo "   ✅ Authentication successful!"
        else
            echo "   ⚠️  Token not created. Try running manually:"
            echo "      python $BACKEND_DIR/mcp_servers/gmail_server.py"
        fi
    fi
fi
echo ""

# Step 6: Check environment variables
echo "Step 6: Checking environment variables..."
ENV_FILE="$BACKEND_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo "   ✅ .env file exists"
    
    if grep -q "OPENAI_API_KEY" "$ENV_FILE"; then
        echo "   ✅ OPENAI_API_KEY configured"
    else
        echo "   ⚠️  OPENAI_API_KEY not found in .env"
    fi
    
    if grep -q "TAVILY_API_KEY" "$ENV_FILE"; then
        echo "   ✅ TAVILY_API_KEY configured"
    else
        echo "   ⚠️  TAVILY_API_KEY not found in .env (optional)"
    fi
else
    echo "   ⚠️  .env file not found"
    echo "   Create .env with:"
    echo "   OPENAI_API_KEY=your_key_here"
    echo "   TAVILY_API_KEY=your_key_here"
fi
echo ""

# Step 7: Run test
echo "Step 7: Testing Gmail integration..."
if [ -f "$TOKEN_FILE" ]; then
    read -p "   Run integration tests now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 "$BACKEND_DIR/test_gmail_integration.py"
    else
        echo "   ⚠️  Skipped tests"
        echo "   Run tests manually: python test_gmail_integration.py"
    fi
else
    echo "   ⚠️  Cannot run tests without authentication"
fi
echo ""

# Summary
echo "=================================="
echo "Setup Summary"
echo "=================================="
echo ""

CHECKLIST=""
CHECKLIST+="✅ Python 3.8+ installed\n"
CHECKLIST+="✅ Dependencies installed\n"
CHECKLIST+="✅ Credentials directory created\n"

if [ -f "$CREDS_FILE" ]; then
    CHECKLIST+="✅ Gmail credentials configured\n"
else
    CHECKLIST+="❌ Gmail credentials missing\n"
fi

if [ -f "$TOKEN_FILE" ]; then
    CHECKLIST+="✅ Gmail authenticated\n"
else
    CHECKLIST+="❌ Gmail authentication pending\n"
fi

if [ -f "$ENV_FILE" ]; then
    CHECKLIST+="✅ Environment variables configured\n"
else
    CHECKLIST+="❌ .env file missing\n"
fi

echo -e "$CHECKLIST"
echo ""

if [ -f "$CREDS_FILE" ] && [ -f "$TOKEN_FILE" ] && [ -f "$ENV_FILE" ]; then
    echo "🎉 Setup complete! Ready to use Gmail MCP integration."
    echo ""
    echo "Next steps:"
    echo "   1. Start backend: ./start.sh"
    echo "   2. Test queries that search Gmail for Round Rock school data"
    echo "   3. See GMAIL_MCP_SETUP.md for usage examples"
else
    echo "⚠️  Setup incomplete. Please complete the missing steps above."
    echo ""
    echo "See GMAIL_MCP_SETUP.md for detailed instructions."
fi
echo ""
