# Facebook Messenger Setup Guide

This guide walks you through setting up Facebook Messenger integration for your RAG bot.

## Prerequisites

- Facebook account (must verify phone number)
- A domain with HTTPS OR ngrok for local testing

---

## Step 1: Create Facebook Developer Account

1. Go to [developers.facebook.com](https://developers.facebook.com/)
2. Click **"My Apps"** → **"Create App"**
3. Select **"Other"** → **"Business"**
4. App Name: `RAG Bot` (or whatever you want)
5. Click **"Create App"**

---

## Step 2: Add Messenger Product

1. Scroll down to "Add products to your app"
2. Find **"Messenger"** and click **"Set up"**
3. You'll be redirected to Messenger settings

---

## Step 3: Get Page Access Token

### If you already have a Facebook Page:
1. In Messenger settings, find **"Access Tokens"**
2. Click **"Add or remove pages"**
3. Select your Facebook Page
4. Click **"Continue"** (allow permissions)
5. The Page Access Token will appear - **copy it**

### If you don't have a Facebook Page:
1. Go to [facebook.com/pages/create](https://www.facebook.com/pages/create)
2. Create a new page (name it whatever)
3. Go back to Developer Console → Messenger → Access Tokens
4. Select your new page
5. Copy the Page Access Token

---

## Step 4: Configure Webhook

1. In Messenger settings, find **"Webhooks"**
2. Click **"Callback URL"**
3. For local testing, first run:
   ```bash
   # Install ngrok (if not installed)
   brew install ngrok  # Mac
   # or download from https://ngrok.com
   
   # Start ngrok
   ngrok http 5000
   ```
4. Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)
5. In Facebook Developer Console:
   - **Callback URL**: `https://YOUR-NGROK-URL/webhook`
   - **Verify Token**: `your_random_verify_token_here` (create your own)
6. Click **"Verify and Save"**

7. Under "Webhooks" → **"Fields"**, subscribe to:
   - `messages`
   - `messaging_postbacks`

8. Under "Webhooks", find your page and click **"Add subscriptions"**

---

## Step 5: Update .env File

1. Edit your `.env` file:
```bash
# Copy from template
cp .env.example .env
```

2. Edit `.env`:
```env
FB_PAGE_ACCESS_TOKEN=YOUR_PAGE_ACCESS_TOKEN_HERE
FB_PAGE_ID=YOUR_PAGE_ID_HERE
FB_VERIFY_TOKEN=your_random_verify_token_here
```

**To get FB_PAGE_ID:**
- Go to your Facebook Page
- Click "About"
- Scroll to "Page ID"

---

## Step 6: Implement send_fb_message()

In `fb_bot.py`, implement the actual Facebook API call:

```python
def send_fb_message(sender_id, message_text):
    import requests
    
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
    
    data = {
        "recipient": {"id": sender_id},
        "message": {"text": message_text}
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        logger.error(f"Failed to send message: {response.text}")
        return False
    
    return True
```

---

## Step 7: Run and Test

1. Start the app:
   ```bash
   python app.py
   ```

2. Make sure ngrok is still running

3. Send a message to your Facebook Page

4. Check the console - you should see:
   ```
   Received from {sender_id}: your message
   Query processed for user {sender_id}: your message...
   [FB SKELETON] Would send to {sender_id}: Bot's response
   ```

---

## Troubleshooting

### "This URL is unreachable"
- Make sure ngrok is running: `ngrok http 5000`
- Check the callback URL matches ngrok URL exactly

### "Verify token mismatch"
- Make sure `FB_VERIFY_TOKEN` in `.env` matches what you entered in Facebook console

### "Page access token is invalid"
- Generate a new token in Messenger → Access Tokens
- Make sure you selected the correct page
- Try removing and re-adding the page

### Messages not reaching webhook
- Check "Webhooks" section → your page should be subscribed
- Under "Fields", make sure `messages` is subscribed

### Bot not responding
- Check console for errors
- Make sure you've uploaded documents and clicked "Update KB"
- Verify `send_fb_message()` is implemented

---

## Important Notes

1. **HTTPS Required** - Facebook only accepts HTTPS webhooks (ngrok provides this)

2. **Token Expiration** - Page access tokens can expire. For production, implement token exchange:
   - Facebook provides long-lived tokens (60 days)
   - Exchange short-lived token for long-lived:
     ```
     https://graph.facebook.com/v18.0/oauth/access_token?
       grant_type=fb_exchange_token&
       client_id=YOUR_APP_ID&
       client_secret=YOUR_APP_SECRET&
       fb_exchange_token=YOUR_SHORT_LIVED_TOKEN
     ```

3. **App Review** - For public use, Facebook requires app review. For testing, your page/messenger works without review.

---

## Production Checklist

- [ ] Deploy with HTTPS (Render, Railway, etc.)
- [ ] Update webhook URL in Facebook console
- [ ] Implement long-lived token exchange
- [ ] Submit for app review (if going public)

---

## Quick Reference

| Item | Where to Find |
|------|---------------|
| App ID | Settings → Basic |
| App Secret | Settings → Basic (show) |
| Page Access Token | Messenger → Access Tokens |
| Page ID | facebook.com/YOUR_PAGE → About → Page ID |
| Verify Token | Your custom string in .env |
