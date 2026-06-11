// In dev, requests go through the Vite proxy → localhost:3333
// In production, set VITE_API_URL to the full API base URL
const BASE = import.meta.env.VITE_API_URL !== 'http://localhost:3333'
  ? (import.meta.env.VITE_API_URL || '')
  : ''

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => req('/health'),

  createChat: (title) =>
    req('/chat/chats', { method: 'POST', body: JSON.stringify({ title }) }),

  listChats: () => req('/chat/chats'),

  getMessages: (chatId) => req(`/chat/chats/${chatId}/messages`),

  sendMessage: (message, chatId, model) =>
    req('/chat/message', {
      method: 'POST',
      body: JSON.stringify({
        message,
        chat_id: chatId || undefined,
        model: model || undefined,
      }),
    }),

  getTrace: (responseId) => req(`/chat/trace/${responseId}`),

  query: (query) =>
    req('/query', { method: 'POST', body: JSON.stringify({ query }) }),

  metadata: () => req('/metadata'),
}
