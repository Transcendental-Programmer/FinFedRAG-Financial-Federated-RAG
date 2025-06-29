# 🚀 Hugging Face Spaces Deployment Guide

## Quick Deploy to HF Spaces (5 minutes)

### Step 1: Prepare Your Repository

Your repository should have these files in the root:
- ✅ `app.py` - Streamlit application
- ✅ `requirements.txt` - Minimal dependencies (streamlit, requests, numpy)
- ✅ `README.md` - With HF Spaces config at the top

### Step 2: Create HF Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - **Owner**: `ArchCoder`
   - **Space name**: `federated-credit-scoring`
   - **Short description**: `Federated Learning Credit Scoring Demo with Privacy-Preserving Model Training`
   - **License**: `MIT`
   - **Space SDK**: `Streamlit` ⚠️ **NOT Docker**
   - **Space hardware**: `Free`
   - **Visibility**: `Public`

### Step 3: Upload Files

**Option A: Direct Upload**
1. Click "Create Space"
2. Upload these files:
   - `app.py`
   - `requirements.txt`

**Option B: Connect GitHub (Recommended)**
1. In Space Settings → "Repository"
2. Connect your GitHub repo
3. Enable "Auto-deploy on push"

### Step 4: Wait for Build

- HF Spaces will install dependencies
- Build your Streamlit app
- Takes 2-3 minutes

### Step 5: Access Your App

Your app will be live at:
```
https://huggingface.co/spaces/ArchCoder/federated-credit-scoring
```

## 🎯 What Users Will See

- **Demo Mode**: Works immediately (no server needed)
- **Interactive Interface**: Enter features, get predictions
- **Educational Content**: Learn about federated learning
- **Professional UI**: Clean, modern design

## 🔧 Troubleshooting

**"Missing app file" error:**
- Ensure `app.py` is in the root directory
- Check that SDK is set to `streamlit` (not docker)

**Build fails:**
- Check `requirements.txt` has minimal dependencies
- Ensure no heavy packages (tensorflow, etc.) in requirements.txt

**App doesn't load:**
- Check logs in HF Spaces
- Verify app.py has no syntax errors

## 📁 Required Files

**`app.py`** (root level):
```python
import streamlit as st
import requests
import numpy as np
import time

st.set_page_config(page_title="Federated Credit Scoring Demo", layout="centered")
# ... rest of your app code
```

**`requirements.txt`** (root level):
```
streamlit
requests
numpy
```

**`README.md`** (with HF config at top):
```yaml
---
title: Federated Credit Scoring
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_port: 8501
tags:
- streamlit
- federated-learning
- machine-learning
- privacy
pinned: false
short_description: Federated Learning Credit Scoring Demo with Privacy-Preserving Model Training
license: mit
---
```

## 🎉 Success!

After deployment, you'll have:
- ✅ Live web app accessible to anyone
- ✅ No server setup required
- ✅ Professional presentation of your project
- ✅ Educational value for visitors

**Your federated learning demo will be live and working!** 🚀 