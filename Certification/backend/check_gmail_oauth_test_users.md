# Gmail OAuth Test Users Setup Guide

## Problem
Your Gmail OAuth app is in "Testing" mode and requires test users to be explicitly added.

## Solution: Add Test Users

### Step 1: Access Google Cloud Console
1. Go to: https://console.cloud.google.com/
2. Sign in with your Google account

### Step 2: Select Your Project
1. Click the project dropdown at the top
2. Select the project where you created Gmail OAuth credentials

### Step 3: Navigate to OAuth Consent Screen
1. Click the hamburger menu (☰) in the top-left
2. Go to **"APIs & Services"** → **"OAuth consent screen"**

### Step 4: Add Test Users
1. Scroll down to the **"Test users"** section
2. Click **"+ ADD USERS"** button
3. Enter email addresses (one per line):
   ```
   nair.vijayan.vipin@gmail.com
   ```
4. Click **"Save"**
5. Click **"Save"** at the bottom of the page

### Step 5: Verify OAuth Credentials
1. Go to **"APIs & Services"** → **"Credentials"**
2. Click on your OAuth 2.0 Client ID
3. Verify:
   - **Authorized redirect URIs** includes: `http://localhost:8000/api/auth/gmail/callback`
   - Application type: Desktop app or Web application

### Step 6: Clear and Reconnect Gmail in Your App
1. In your School Assistant app, go to Settings
2. Click **"Disconnect Gmail"** (if connected)
3. **Sign out** completely from the app
4. **Sign in again** with Gmail
5. This will trigger a fresh OAuth flow with the test user permissions

## Alternative: Publish Your App (Not Recommended for Testing)

If you want to skip the test user requirement:
1. Go to OAuth consent screen
2. Click **"PUBLISH APP"**
3. Confirm the publishing
4. ⚠️ **Warning**: This makes your app available to anyone with a Google account

## Troubleshooting

### Error: "Access blocked: Authorization Error"
- **Cause**: Your email is not in the test users list
- **Solution**: Add your email to test users (Step 4 above)

### Error: "This app isn't verified"
- **Cause**: App is in testing mode
- **Solution**: Click "Advanced" → "Go to [App Name] (unsafe)" during OAuth flow

### Error: "The OAuth client was not found"
- **Cause**: Wrong OAuth credentials or project
- **Solution**: Verify you're using the correct `gmail_credentials.json` file

## After Adding Test User

1. The OAuth flow should work without the "access blocked" error
2. You'll still see "This app isn't verified" warning - this is normal for testing
3. Click "Advanced" → "Continue to [App Name]" to proceed

## Current User to Add
- **Email**: nair.vijayan.vipin@gmail.com
- **Purpose**: Primary test user for Gmail integration

## Next Steps After Adding Test User
1. Clear browser cache (optional but recommended)
2. Sign out from School Assistant app
3. Sign in again with Gmail
4. Grant Gmail permissions when prompted
5. Test the query: "tell me about the school programs this year"
