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
const selectedModel = ref('')
const showMeta = ref(false)

// ── Computed ──────────────────────────────────────────────────────────────────
const activeTitle = computed(() => {
  const c = chats.value.find(c => c.chat_id === activeChatId.value)
  return c?.title || ''
})

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

async function loadChats() {
  loadingChats.value = true
  try {
    const res = await api.listChats()
    chats.value = res.chats || []
  } catch {
    chats.value = []
  } finally {
    loadingChats.value = false
  }
}

// ── Chat management ───────────────────────────────────────────────────────────
async function createChat() {
  try {
    const res = await api.createChat('Novo chat')
    await loadChats()
    selectChatById(res.chat_id)
  } catch (e) {
    console.error('Erro ao criar chat:', e)
  }
}

function selectChat(chat) {
  selectChatById(chat.chat_id)
}

async function selectChatById(chatId) {
  activeChatId.value = chatId
  messages.value = []
  sendError.value = null
  try {
    const res = await api.getMessages(chatId)
    messages.value = (res.messages || []).map(m => ({
      id: m.message_id,
      role: m.role,
      content: m.content,
      created_at: m.created_at,
    }))
  } catch {
    messages.value = []
  }
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text.trim() || sending.value) return

  sendError.value = null

  // If no chat selected, create one automatically
  if (!activeChatId.value) {
    try {
      const res = await api.createChat(text.slice(0, 50))
      await loadChats()
      activeChatId.value = res.chat_id
    } catch (e) {
      sendError.value = 'Erro ao criar sessão: ' + e.message
      return
    }
  }

  // Optimistic user message
  const tempId = 'tmp-' + Date.now()
  messages.value.push({ id: tempId, role: 'user', content: text })
  sending.value = true

  try {
    const res = await api.sendMessage(text, activeChatId.value, selectedModel.value || undefined)

    // Remove optimistic
    messages.value = messages.value.filter(m => m.id !== tempId)

    // Add user + assistant
    messages.value.push({ id: 'u-' + res.response_id, role: 'user', content: text })
    messages.value.push({
      id: res.response_id,
      role: 'assistant',
      content: res.answer,
      search_results: res.search_results,
      agent_thoughts: res.agent_thoughts,
      response_time_seconds: res.response_time_seconds,
      confidence_score: res.confidence_score,
      model: res.model,
      tokens: res.tokens,
    })

    // Update active chat title if first message
    const chat = chats.value.find(c => c.chat_id === activeChatId.value)
    if (chat && (chat.message_count === 0 || chat.title === 'Novo chat')) {
      await loadChats()
    } else {
      // Just update count
      await loadChats()
    }
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
