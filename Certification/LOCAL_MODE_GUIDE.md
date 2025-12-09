# Local Mode Configuration Guide

This guide explains how to run the School Events RAG Application in local mode on your localhost.

## Overview

The application supports two modes:
- **Production Mode**: Connects to the production server (Railway/Vercel)
- **Local Mode**: Runs entirely on localhost (recommended for development)

## Quick Start - Local Mode

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd Certification/backend
   ```

2. Copy the example environment file (if you don't have a .env file):
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and set LOCAL_MODE to true:
   ```bash
   LOCAL_MODE=true
   ```

4. Add your API keys to `.env`:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   TAVILY_API_KEY=your-tavily-api-key
   # ... other keys as needed
   ```

5. Start the backend server:
   ```bash
   python main.py
   ```
   
   The backend will run on `http://localhost:8000`

### 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd Certification/frontend
   ```

2. Copy the local environment file:
   ```bash
   cp .env.local .env
   ```
   
   This file already has `REACT_APP_LOCAL_MODE=true` set.

3. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

4. Start the frontend:
   ```bash
   npm start
   ```
   
   The frontend will run on `http://localhost:3000`

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

## Switching Between Modes

### To Switch to Local Mode:

**Backend:**
```bash
# In backend/.env
LOCAL_MODE=true
```

**Frontend:**
```bash
# In frontend/.env
REACT_APP_LOCAL_MODE=true
```

### To Switch to Production Mode:

**Backend:**
```bash
# In backend/.env
LOCAL_MODE=false
```

**Frontend:**
```bash
# In frontend/.env
REACT_APP_LOCAL_MODE=false
```

## What Changes in Local Mode?

### Backend Changes:
- ✅ CORS is configured for `http://localhost:3000` only
- ✅ Vercel production URL patterns are disabled
- ✅ Logs indicate "Running in LOCAL MODE"

### Frontend Changes:
- ✅ API calls are directed to `http://localhost:8000`
- ✅ No calls to production servers

## Troubleshooting

### Frontend can't connect to backend
1. Ensure backend is running on port 8000
2. Check that `REACT_APP_LOCAL_MODE=true` in frontend/.env
3. Verify no CORS errors in browser console

### Backend CORS errors
1. Ensure `LOCAL_MODE=true` in backend/.env
2. Restart the backend server after changing .env
3. Clear browser cache and reload

### Environment variables not working
1. Restart the development servers after changing .env files
2. For React, environment variables must start with `REACT_APP_`
3. Check for typos in variable names

## Production Deployment

When deploying to production:

1. Set `LOCAL_MODE=false` in backend environment
2. Set `REACT_APP_LOCAL_MODE=false` in frontend environment
3. Ensure production URLs are configured correctly

## Environment Files Summary

| File | Purpose | Key Variable |
|------|---------|--------------|
| `backend/.env` | Backend configuration | `LOCAL_MODE=true/false` |
| `backend/.env.example` | Backend template | Reference only |
| `frontend/.env.local` | Local development | `REACT_APP_LOCAL_MODE=true` |
| `frontend/.env.example` | Frontend template | Reference only |

## Gmail OAuth Setup for Local Mode

If using Gmail integration, you need to configure the OAuth redirect URI:

1. **Go to Google Cloud Console:**
   - https://console.cloud.google.com/apis/credentials

2. **Add Local Redirect URI:**
   - Select your OAuth 2.0 Client ID
   - Add to "Authorized redirect URIs":
     ```
     http://localhost:8000/api/auth/gmail/callback
     ```
   - Click SAVE and wait 5 minutes

3. **See Complete Guide:**
   - [FIX_OAUTH_REDIRECT_URI.md](FIX_OAUTH_REDIRECT_URI.md) for detailed instructions

## Tips

- Use `.env.local` for local development (git-ignored)
- Never commit API keys to version control
- Keep `.env.example` files updated with new variables
- Test in local mode before deploying to production
- Add both local and production OAuth redirect URIs to Google Console for flexibility
