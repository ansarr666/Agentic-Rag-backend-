# HTTP API contract

Backend base URL is supplied to the frontend by `VITE_API_BASE_URL` or `/config.js` `backendUrl`. All JSON POSTs use `Content-Type: application/json`. Responses are JSON unless the OAuth authorize endpoint redirects. Failed requests use an HTTP error status and `{"error": "..."}` (some 500s include `request_id`); `429` indicates a rate limit. CORS permits exact origins in backend `ALLOWED_ORIGINS`.

## Endpoints called by the frontend

| Method | Path | Request | Success response | Auth |
| --- | --- | --- | --- | --- |
| POST | `/api/public/lead` | JSON `name,email,phone,country_code,company?` | `success,lead_id,conversation_id` | Public |
| POST | `/api/public/conversation` | JSON `question,history:[{role,content}],lead_id,conversation_id,name` | `answer,suggested_questions, intent` | Public |
| POST | `/api/public/intent` | JSON `lead_id,conversation_id,phone,requested_action,summary` | `success,message` | Public |
| POST | `/api/query` | JSON `question,user_role,user_groups?` | Structured RAG response with answer, sources, decision, evidence, trace | `X-API-Key` or matching bearer key when configured |
| POST | `/api/documents/upload` | `multipart/form-data`, one or more `file` parts; txt, pdf, md, docx, html, htm | `success,uploaded,total_documents,total_chunks` | Same as query |
| GET | `/api/integrations/google-drive/status` | None | Connector status JSON | Same as query |
| POST | `/api/integrations/google-drive/connect` | JSON `account,workspace` | Connector status JSON | Admin key |
| POST | `/api/integrations/google-drive/disconnect` | None | Connector status JSON | Admin key |
| GET | `/api/integrations/google-drive/sources` | None | Available source JSON | Admin key |
| POST | `/api/integrations/google-drive/sources` | JSON `selected_sources` | Connector status JSON | Admin key |
| POST | `/api/integrations/google-drive/sync` | None | Sync summary JSON | Admin key |
| GET | `/api/evaluations/latest` | None | Decision benchmark report JSON | Same as query |
| POST | `/api/evaluations/run` | None | Decision benchmark report JSON | Same as query |
| POST | `/api/trace/query` | JSON `question,user_role` | `query,decision,evidence_state,tools_used,confidence_*,latency_ms,execution_trace,evidence_signals,sources,answer` | Same as query |

The browser sends `X-API-Key` for admin requests after the operator enters it. The visitor endpoints send no credential. Upload requests let the browser set the multipart boundary and therefore do not set `Content-Type` manually. Validation errors are normally `400`, invalid admin credentials `401`, unexpected failures `500`, and rate limits `429`.

## Other existing backend endpoints

| Method | Path | Use |
| --- | --- | --- |
| GET | `/api/health` | Health, provider, uptime, connector status |
| GET | `/api/auth/status` | API key configuration and request auth status |
| POST | `/api/public/chat` | Stateless prospect query: `question` to `answer,suggested_queries` |
| GET | `/api/admin/leads` | Admin key; `{"leads":[...]}` |
| GET | `/api/integrations/google-drive/authorize` | Redirect browser to Google consent; backend `GOOGLE_REDIRECT_URI` |
| GET | `/api/integrations/google-drive/callback` | Google calls backend with `code` or `error`; success `{"connected":true}` |
| GET | `/api/tools` | Tool definitions |
| POST | `/api/tools/<tool_id>/toggle` | Admin key; `success,tool` |
| GET, POST | `/api/evaluations/retrieval` | Retrieval benchmark report |
