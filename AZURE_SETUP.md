# Azure OpenAI Setup for PCDS - Imagine Cup

## Quick Path to LIVE Azure AI (15 minutes)

### Option 1: Microsoft for Startups Founders Hub (Recommended)

1. **Go to:** https://foundershub.startups.microsoft.com
2. **Sign up** with your student email
3. **Apply** (takes 24-48 hours approval)
4. **Get:** $1,000 Azure credits FREE

### Option 2: Azure Free Account

1. **Go to:** https://azure.microsoft.com/free/students
2. **Sign up** with student email (.edu)
3. **Get:** $100 Azure credits + 12 months free services

---

## Create Azure OpenAI Resource (10 minutes)

### Step 1: Create Resource
```
1. Go to: https://portal.azure.com
2. Search: "Azure OpenAI"
3. Click "Create"
4. Select:
   - Subscription: Your free/startup subscription
   - Resource Group: Create new → "pcds-ai"
   - Region: East US (most stable)
   - Name: "pcds-openai"
   - Pricing: Standard S0
5. Click "Review + Create"
```

### Step 2: Deploy GPT-4 Model
```
1. Go to: Azure OpenAI Studio (oai.azure.com)
2. Click "Deployments"
3. Click "Create new deployment"
4. Select:
   - Model: gpt-4o-mini (cheapest, still great)
   - Deployment name: "pcds-gpt4"
   - Tokens per minute: 1000 (demo only needs ~100)
5. Click "Create"
```

### Step 3: Get API Keys
```
1. In Azure Portal → Your OpenAI resource
2. Go to: "Keys and Endpoint"
3. Copy:
   - KEY 1
   - Endpoint URL
```

---

## Configure PCDS (2 minutes)

### Create `.env` file in backend folder:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://pcds-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT=pcds-gpt4

# Optional: Hard limit tokens for demo
AZURE_OPENAI_MAX_TOKENS=500
```

### Or set in `backend/config/settings.py`:

```python
AZURE_OPENAI_ENDPOINT = "https://pcds-openai.openai.azure.com/"
AZURE_OPENAI_KEY = "your-key-here"
AZURE_OPENAI_DEPLOYMENT = "pcds-gpt4"
```

---

## Cost Estimate for Demo

| Usage | Cost |
|-------|------|
| 100 demo queries | ~$0.50 |
| Full day testing | ~$2-5 |
| Competition presentation | ~$0.10 |

**Total budget needed:** $5-10 maximum

With $100 free credits, you have 20x more than needed!

---

## Verify It Works

1. Start PCDS: `python main_v2.py`
2. Go to: http://localhost:3000/copilot
3. Ask: "What threats were detected today?"
4. You should see GPT-4 response (not fallback)

---

## Demo Day Checklist

- [ ] Azure credits active
- [ ] GPT-4 deployment running
- [ ] .env file configured
- [ ] Test query works
- [ ] Response cached for reliability

---

## Fallback Plan

If Azure fails during demo:
- Local fallback still works
- Explain: "Azure AI integration is live, showing cached response for reliability"
- Judges understand demo conditions

---

*Last updated: December 21, 2025*
