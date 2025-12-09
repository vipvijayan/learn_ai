# Fixing Google OAuth Redirect URI Mismatch Error

## Error Message
```
Error 400: redirect_uri_mismatch
You can't sign in because School Assistant sent an invalid request.
```

## Problem
The redirect URI configured in Google Cloud Console doesn't match the one your application is sending.

## Solution

### Step 1: Check Your Current Mode
Your application is running in **LOCAL MODE** (localhost). The redirect URI being used is:
```
http://localhost:8000/api/auth/gmail/callback
```

### Step 2: Add Redirect URI to Google Cloud Console

1. **Go to Google Cloud Console:**
   - Visit: https://console.cloud.google.com/apis/credentials
   - Select your project

2. **Find Your OAuth 2.0 Client ID:**
   - Click on your OAuth 2.0 Client ID (the one you're using)

3. **Add Authorized Redirect URI:**
   - Scroll to "Authorized redirect URIs"
   - Click "ADD URI"
   - Add: `http://localhost:8000/api/auth/gmail/callback`
   - Click "SAVE"

4. **Wait a Few Minutes:**
   - Google needs time to propagate the changes (usually 1-5 minutes)

### Step 3: Test Again
After saving and waiting a few minutes, try signing in with Google again.

## For Production Deployment

When deploying to production (with `LOCAL_MODE=false`), also add:
```
https://school-assistant-production.up.railway.app/api/auth/gmail/callback
```

## Complete Authorized Redirect URIs List

For full flexibility, add both URIs to your Google Cloud Console:

```
http://localhost:8000/api/auth/gmail/callback
https://school-assistant-production.up.railway.app/api/auth/gmail/callback
```

This allows the app to work in both local and production modes.

## Visual Guide

### Google Cloud Console Setup:
```
1. Select Project → "School Assistant" (or your project name)
2. APIs & Services → Credentials
3. Click your OAuth 2.0 Client ID
4. Under "Authorized redirect URIs":
   ┌─────────────────────────────────────────────────────┐
   │ http://localhost:8000/api/auth/gmail/callback      │ [X]
   │ https://school-assistant-production.up.railway...  │ [X]
   │                                                     │
   │ [+ ADD URI]                                         │
   └─────────────────────────────────────────────────────┘
5. [SAVE]
```

## Troubleshooting

### Still Getting Error?
1. **Clear browser cache** and try again
2. **Wait 5 minutes** after saving in Google Console
3. **Check the exact URL** in the error message matches what you added
4. **Verify LOCAL_MODE** is set correctly in `backend/.env`

### Different Port?
If your backend is running on a different port (not 8000):
- Update the redirect URI in Google Console
- Or set `GMAIL_REDIRECT_URI` in `.env`:
  ```bash
  GMAIL_REDIRECT_URI=http://localhost:YOUR_PORT/api/auth/gmail/callback
  ```

### Using a Custom Domain?
Set the redirect URI in your `.env`:
```bash
GMAIL_REDIRECT_URI=https://your-domain.com/api/auth/gmail/callback
```

## Quick Test

After adding the redirect URI, test with this command in your backend directory:
```bash
python check_oauth_config.py
```

This will verify your OAuth configuration is correct.

## Need Help?

Check the logs when starting the backend. You should see:
```
🏠 Gmail OAuth redirect URI (LOCAL): http://localhost:8000/api/auth/gmail/callback
```

If you see a different URL, that's what needs to be in Google Cloud Console.
