# 🚀 Quick Reference: Local Mode

## Start Local Development

### Terminal 1 - Backend
```bash
cd Certification/backend
# Make sure LOCAL_MODE=true in .env
python main.py
```

### Terminal 2 - Frontend  
```bash
cd Certification/frontend
# Make sure REACT_APP_LOCAL_MODE=true in .env.local or .env
npm start
```

### Access
```
http://localhost:3000
```

## Environment Variable Quick Reference

| Location | File | Variable | Value for Local | Value for Production |
|----------|------|----------|-----------------|---------------------|
| Backend | `.env` | `LOCAL_MODE` | `true` | `false` |
| Frontend | `.env.local` or `.env` | `REACT_APP_LOCAL_MODE` | `true` | `false` |

## Mode Indicators

**Backend Console:**
- Local: `🏠 Running in LOCAL MODE`
- Production: `🌐 Running in PRODUCTION MODE`

**Frontend API Calls:**
- Local: `http://localhost:8000/*`
- Production: `https://school-assistant-production.up.railway.app/*`

## Common Issues

### "CORS Error" in Browser
- ✅ Check `LOCAL_MODE=true` in backend/.env
- ✅ Restart backend server
- ✅ Clear browser cache

### "Failed to fetch" Error
- ✅ Check backend is running on port 8000
- ✅ Check `REACT_APP_LOCAL_MODE=true` in frontend
- ✅ Restart frontend dev server

### Environment Variables Not Working
- ✅ Restart both servers after changing .env
- ✅ Check variable names (must start with `REACT_APP_` in frontend)

## Files to Configure

```
Certification/
├── backend/
│   └── .env                    ← Set LOCAL_MODE=true
└── frontend/
    └── .env.local              ← Already set to local mode
```

## Switch to Production

**Backend (.env):**
```bash
LOCAL_MODE=false
```

**Frontend (.env):**
```bash
REACT_APP_LOCAL_MODE=false
```

Then restart both servers.

---

📖 **Full Documentation:** See [LOCAL_MODE_GUIDE.md](LOCAL_MODE_GUIDE.md)
