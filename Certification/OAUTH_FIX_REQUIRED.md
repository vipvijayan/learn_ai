# 🔧 Google OAuth Fix - Action Required

## ⚠️ IMMEDIATE ACTION NEEDED

You're getting the `redirect_uri_mismatch` error because your Google Cloud Console isn't configured for local development.

## 🎯 Quick Fix (5 minutes)

### 1. Open Google Cloud Console
Go to: https://console.cloud.google.com/apis/credentials

### 2. Find Your OAuth Client
- Click on your OAuth 2.0 Client ID
- (The one you're currently using for this project)

### 3. Add This Redirect URI
In the "Authorized redirect URIs" section, add:
```
http://localhost:8000/api/auth/gmail/callback
```

### 4. Save and Wait
- Click **SAVE**
- Wait **5 minutes** for changes to propagate
- Then try signing in again

## ✅ What Was Fixed in Code

The backend code now automatically uses the correct redirect URI based on LOCAL_MODE:

| Mode | Redirect URI |
|------|-------------|
| **Local** (`LOCAL_MODE=true`) | `http://localhost:8000/api/auth/gmail/callback` |
| **Production** (`LOCAL_MODE=false`) | `https://school-assistant-production.up.railway.app/api/auth/gmail/callback` |

## 📝 Recommended: Add Both URIs

For maximum flexibility, add **both** URIs to Google Cloud Console:

```
http://localhost:8000/api/auth/gmail/callback
https://school-assistant-production.up.railway.app/api/auth/gmail/callback
```

This way, your app works in both environments without changing Google settings.

## 🔍 Verify Your Setup

After starting the backend, check the logs. You should see:
```
🏠 Gmail OAuth redirect URI (LOCAL): http://localhost:8000/api/auth/gmail/callback
```

If you see this, that's the exact URL you need to add to Google Cloud Console.

## 📚 Need More Help?

See [FIX_OAUTH_REDIRECT_URI.md](FIX_OAUTH_REDIRECT_URI.md) for:
- Detailed step-by-step guide with screenshots
- Troubleshooting tips
- Alternative solutions

## 🧪 Test After Setup

1. Restart your backend server
2. Check the logs for the redirect URI
3. Try signing in with Google
4. Should work now! ✅

---

**Remember:** Any time you switch between local and production mode, make sure the corresponding redirect URI is in Google Cloud Console!
