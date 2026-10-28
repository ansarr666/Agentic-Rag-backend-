# Client Onboarding & Implementation Guide
## Enterprise Conversational AI & Lead Automation Platform

---

### Welcome to Your Custom AI Solution

Thank you for partnering with us to deploy your custom **Enterprise Conversational AI Platform**. This solution is engineered to act as a 24/7 technical and sales consultant on your website—answering customer inquiries with **zero hallucination**, verifying qualified leads via **one-time password (OTP)**, and booking consultations directly into your sales team’s calendar.

This document outlines:
1. **What we need from you (The Requirements Checklist)**
2. **The 3 to 5 business day implementation timeline**
3. **How the simple 1-line website integration works**
4. **Client Intake Form (to fill out and return)**

---

## 1. What We Need From You (Client Checklist)

To ensure an accurate and rapid launch, please provide the assets listed below:

### A. Knowledge Base & Company Documentation
* [ ] **Service & Product Literature**: PDF brochures, product catalogs, service rate cards, case studies, or whitepapers you want the AI to reference.
* [ ] **Frequently Asked Questions (FAQs)**: Any existing customer questions, standard policies, or onboarding manuals.
* [ ] **Out-of-Scope / Excluded Topics**: Specific topics the AI must **refuse** to answer (e.g., custom enterprise pricing, internal salaries, confidential contracts, or legal advice).

### B. Branding & Visual Identity
* [ ] **Logo / Icon**: Transparent `.png` or `.svg` (minimum 256x256 px).
* [ ] **Brand Colors**: Your primary brand hex code (e.g., `#2563eb`) and secondary accent color.
* [ ] **Assistant Persona**: The name of your AI consultant (e.g., *"Apex Advisor"*, *"Orion AI"*) and preferred tone (e.g., *Professional & Consultative* or *Friendly & Casual*).
* [ ] **Starter Prompts**: 3 to 4 quick-select question chips you want first-time visitors to see (e.g., *"What services do you offer?"*, *"Request a quote"*, *"Book a discovery call"*).

### C. Booking & Sales Team Preferences
* [ ] **Consultation Duration**: Standard meeting length (e.g., 15, 30, or 45 minutes).
* [ ] **Meeting Format**: Phone call, Google Meet, Zoom link, or custom scheduling URL (Calendly / Cal.com).
* [ ] **Lead Notification Recipient**: Email address(es) where verified prospect inquiries and call bookings should be dispatched immediately.
* [ ] **Sending Identity**: Authorized sender email address (e.g., `consultations@yourcompany.com`) or domain verification approval so OTPs and calendar invites arrive under your brand name.

### D. Website & Security Details
* [ ] **Target Website Domain**: The exact public website domain (e.g., `https://yourcompany.com`) where the widget will live (required to configure security and CORS whitelisting).
* [ ] **Platform**: WordPress, Webflow, Shopify, Squarespace, Wix, or custom React/HTML.

---

## 2. Implementation Timeline (3 to 5 Business Days)

Because our underlying multi-path retrieval engine is already containerized, our standard onboarding from asset receipt to live launch takes **3 to 5 business days**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Day 1: Asset Intake & Environment Provisioning                              │
│ Day 2: Document Ingestion & Hybrid Vector Indexing                          │
│ Day 3: Custom Guardrail Tuning & Rule 6 Factual Benchmarking                │
│ Day 4: Private Staging Sandbox Review (Client UAT)                          │
│ Day 5: Production Embed & Live Verification                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Phase | Milestone | Deliverables & Actions |
| :--- | :--- | :--- |
| **Day 1** | **Asset Intake & Setup** | We receive your documentation, brand assets, and scheduling preferences. We provision your isolated secure cloud container and database. |
| **Day 2** | **Ingestion & Dual Indexing** | Your documents are recursively parsed, chunked, and dual-indexed into dense vector and lexical BM25 indices with cross-encoder reranking. |
| **Day 3** | **Guardrail & Safety Tuning** | We configure your out-of-scope rules, anti-exfiltration boundaries, and deterministic math handlers. We run automated evaluation benchmarks to confirm zero hallucination. |
| **Day 4** | **Staging Review (Client UAT)** | We provide a private, interactive staging link (`https://demo.yourcompany.ai`) for your team to test conversational quality, OTP verification, and calendar booking. |
| **Day 5** | **Live Launch & Verification** | You paste a single 1-line script tag on your live website. We verify SSL, CORS whitelisting, live email delivery, and 1-click Google/Outlook calendar synchronization. |

---

## 3. How the Website Integration Works

Integration requires **zero backend development** from your web team. It is embedded using a lightweight, asynchronous 1-line script tag.

### Option 1: Standard Script Tag
Paste this tag right before the closing `</body>` tag on your website (or within Google Tag Manager):

```html
<!-- Enterprise AI Assistant Widget -->
<script src="https://chat.yourdomain.com/widget-loader.js" async></script>
```

### Option 2: Dedicated Full-Page Portal / iFrame
If you want the assistant accessible on a dedicated page (e.g., `yourcompany.com/assistant`):
```html
<iframe 
  src="https://chat.yourdomain.com/widget" 
  style="width: 100%; height: 750px; border: none; border-radius: 16px;" 
  title="AI Consultant">
</iframe>
```

> [!NOTE]
> **Performance Guarantee**: The script is asynchronous and sandboxed in an isolated container. It adds zero render-blocking overhead and will never affect your website's layout, fonts, or SEO speed score.

---

## 4. Security, Privacy & Compliance Guarantees

* **Zero Hallucination ("Rule 6" Contract)**: If an answer is not backed by evidence in your approved documents, the assistant explicitly declines to answer and routes the visitor to book a call with your human team.
* **Anti-Exfiltration Shield**: Internal keywords, proprietary data, and sensitive policies are strictly blocked from external public search tools.
* **Verified Lead Qualification**: Automated screening against 30+ disposable email domains, RFC-compliant syntax checking, and instant sales leadership dispatch.
* **Full Data Ownership**: Your documentation and visitor records are strictly isolated. Your data is **never** used to train public foundation models.

---

## 5. Client Intake Form (Fill Out & Return)

Please complete the fields below and return this form along with your document files:

```text
================================================================================
CLIENT ONBOARDING INTAKE FORM
================================================================================

1. GENERAL INFORMATION
   Company Name: ______________________________________________________________
   Primary Contact: ___________________________________________________________
   Contact Email: _____________________________________________________________
   Target Website URL: ________________________________________________________

2. BRANDING PREFERENCES
   Assistant Name: ____________________________________________________________
   Brand Hex Color (e.g. #2563eb): ____________________________________________
   Logo Attached: [ ] Yes   [ ] No (Use default)
   Greeting Headline: _________________________________________________________

3. SUGGESTED STARTER PROMPTS (3-4 quick-click questions)
   1. _________________________________________________________________________
   2. _________________________________________________________________________
   3. _________________________________________________________________________
   4. _________________________________________________________________________

4. LEAD & CALENDAR PREFERENCES
   Sales Notification Email(s): _______________________________________________
   Consultation Duration: [ ] 15 mins   [ ] 30 mins   [ ] 45 mins
   Preferred Meeting Link / Tool: _____________________________________________
   Specific Out-of-Scope Topics (What NOT to answer):
   - __________________________________________________________________________
   - __________________________________________________________________________
   - __________________________________________________________________________

================================================================================
```
