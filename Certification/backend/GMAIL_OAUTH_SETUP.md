# Gmail OAuth Setup Instructions

The application needs Google OAuth credentials to connect to Gmail. You have two options:

## Option 1: Using Credentials File (Recommended)

1. **Get OAuth Credentials from Google Cloud Console:**
   - Go to https://console.cloud.google.com/
   - Create a new project or select existing one
   - Enable Gmail API
   - Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
   - Application type: "Web application"
   - Authorized redirect URIs: `http://localhost:8000/api/auth/gmail/callback`
   - Download the JSON file

2. **Set up the credentials:**
   ```bash
   cd /Users/vipinvijayan/Developer/projects/AI/AIMakerSpace/code/learn_ai_0/Certification/backend
   mkdir -p credentials
   # Copy your downloaded JSON file to credentials/gmail_credentials.json
   ```

3. **The JSON file should look like:**
   ```json
   {
     "web": {
       "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
       "project_id": "your-project",
       "auth_uri": "https://accounts.google.com/o/oauth2/auth",
       "token_uri": "https://oauth2.googleapis.com/token",
       "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
       "client_secret": "YOUR_CLIENT_SECRET",
       "redirect_uris": ["http://localhost:8000/api/auth/gmail/callback"]
     }
   }
   ```

## Option 2: Using Environment Variables (Quick Setup)

Add these to your `.env` file:

```bash
GOOGLE_CLIENT_ID=your_client_id_here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret_here
```

Then restart the backend server.

## Verification

After setup, try to connect Gmail from the frontend. The OAuth flow should work without the "500: Gmail OAuth credentials not configured" error.

## Security Note

**NEVER commit credentials to git!** The `credentials/` folder is already in `.gitignore`.
