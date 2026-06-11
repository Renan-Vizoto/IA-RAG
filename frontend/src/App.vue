<template>
  <div class="app-layout">
    <SidebarPanel
      :chats="chats"
      :activeChatId="activeChatId"
      :loadingChats="loadingChats"
      :online="online"
      :selectedModel="selectedModel"
      @new-chat="createChat"
      @select-chat="selectChat"
      @change-model="selectedModel = $event"
    />

    <ChatView
      :activeChatId="activeChatId"
      :activeTitle="activeTitle"
      :messages="messages"
      :loading="sending"
      :error="sendError"
      :showMeta="showMeta"
      @send="sendMessage"
      @toggle-meta="showMeta = !showMeta"
    />

    <MetadataPanel
      :visible="showMeta"
      @close="showMeta = false"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import SidebarPanel from './components/SidebarPanel.vue'
import ChatView from './components/ChatView.vue'
import MetadataPanel from './components/MetadataPanel.vue'
import { api } from './api.js'

// ── State ─────────────────────────────────────────────────────────────────────
const chats = ref([])
const activeChatId = ref(null)
const messages = ref([])
const sending = ref(false)
const sendError = ref(null)
const loadingChats = ref(false)
const online = ref(false)
const selectedModel = ref('qwen3.5-2b-unsloth')
const showMeta = ref(false)

let messagesLoadSeq = 0

// ── Computed ──────────────────────────────────────────────────────────────────
const activeTitle = computed(() => {
  const c = chats.value.find(c => c.chat_id === activeChatId.value)
  return c?.title || ''
})

function mapMessages(rows) {
  return (rows || []).map(m => ({
    id: m.message_id,
    role: m.role,
    content: m.content,
    created_at: m.created_at,
  }))
}

function enrichLastAssistant(res) {
  const next = [...messages.value]
  for (let i = next.length - 1; i >= 0; i -= 1) {
    if (next[i].role !== 'assistant') continue
    next[i] = {
      ...next[i],
      search_results: res.search_results ?? [],
      agent_thoughts: res.agent_thoughts ?? '',
      response_time_seconds: res.response_time_seconds,
      confidence_score: res.confidence_score,
      model: res.model,
      tokens: res.tokens,
    }
    messages.value = next
    break
  }
}

function buildAssistantMessage(res, text) {
  return [
    { id: 'u-' + res.response_id, role: 'user', content: text },
    {
      id: res.response_id,
      role: 'assistant',
      content: res.answer ?? '',
      search_results: res.search_results ?? [],
      agent_thoughts: res.agent_thoughts ?? '',
      response_time_seconds: res.response_time_seconds,
      confidence_score: res.confidence_score,
      model: res.model,
      tokens: res.tokens,
    },
  ]
}

function patchChatSummary(chatId, patch) {
  const idx = chats.value.findIndex(c => c.chat_id === chatId)
  if (idx >= 0) {
    const updated = { ...chats.value[idx], ...patch }
    chats.value = [
      ...chats.value.slice(0, idx),
      updated,
      ...chats.value.slice(idx + 1),
    ]
    return
  }

  chats.value = [{
    chat_id: chatId,
    title: patch.title || 'Chat',
    message_count: patch.message_count ?? 0,
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    created_at: patch.created_at || new Date().toISOString(),
    updated_at: patch.updated_at || new Date().toISOString(),
  }, ...chats.value]
}

async function loadMessagesForChat(chatId) {
  const seq = ++messagesLoadSeq
  const res = await api.getMessages(chatId)
  if (seq !== messagesLoadSeq || activeChatId.value !== chatId) {
    return null
  }
  messages.value = mapMessages(res.messages)
  return messages.value
}

// ── Init ──────────────────────────────────────────────────────────────────────
onMounted(async () => {
  await checkHealth()
  await loadChats()
})

async function checkHealth() {
  try {
    await api.health()
    online.value = true
  } catch {
    online.value = false
  }
}

async function loadChats({ silent = false } = {}) {
  if (!silent) loadingChats.value = true
  try {
    const res = await api.listChats()
    chats.value = res.chats || []
  } catch {
    chats.value = []
  } finally {
    if (!silent) loadingChats.value = false
  }
}

// ── Chat management ───────────────────────────────────────────────────────────
async function createChat() {
  try {
    const res = await api.createChat('Novo chat')
    await loadChats({ silent: true })
    await selectChatById(res.chat_id)
  } catch (e) {
    console.error('Erro ao criar chat:', e)
  }
}

function selectChat(chat) {
  selectChatById(chat.chat_id)
}

async function selectChatById(chatId) {
  if (sending.value && chatId === activeChatId.value) return

  const previousChatId = activeChatId.value
  activeChatId.value = chatId
  sendError.value = null

  if (previousChatId !== chatId) {
    messages.value = []
  }

  try {
    await loadMessagesForChat(chatId)
  } catch {
    if (activeChatId.value === chatId) {
      messages.value = []
    }
  }
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text.trim() || sending.value) return

  sendError.value = null

  if (!activeChatId.value) {
    try {
      const res = await api.createChat(text.slice(0, 50))
      activeChatId.value = res.chat_id
      patchChatSummary(res.chat_id, {
        title: res.title,
        message_count: 0,
        created_at: res.created_at,
        updated_at: res.created_at,
      })
      await loadChats({ silent: true })
    } catch (e) {
      sendError.value = 'Erro ao criar sessão: ' + e.message
      return
    }
  }

  const chatId = activeChatId.value
  const tempId = 'tmp-' + Date.now()
  messages.value.push({ id: tempId, role: 'user', content: text })
  sending.value = true

  try {
    const res = await api.sendMessage(text, chatId, selectedModel.value || undefined)
    const targetChatId = res.chat_id || chatId

    if (targetChatId !== activeChatId.value) {
      activeChatId.value = targetChatId
    }

    messages.value = messages.value.filter(m => m.id !== tempId)
    try {
      const loaded = await loadMessagesForChat(targetChatId)
      if (loaded) {
        enrichLastAssistant(res)
      } else {
        messages.value = [...messages.value, ...buildAssistantMessage(res, text)]
      }
    } catch {
      messages.value = [...messages.value, ...buildAssistantMessage(res, text)]
    }
    await loadChats({ silent: true })
    patchChatSummary(targetChatId, {
      title: res.title || text.slice(0, 50),
      message_count: res.message_count,
      updated_at: new Date().toISOString(),
    })
  } catch (e) {
    messages.value = messages.value.filter(m => m.id !== tempId)
    sendError.value = e.message
  } finally {
    sending.value = false
  }
}
</script>

<style>
html, body {
  height: 100%;
  overflow: hidden;
}
#app {
  height: 100%;
  display: flex;
}
.app-layout {
  flex: 1;
  display: flex;
  height: 100%;
  overflow: hidden;
  background: var(--bg);
}
</style>
