<template>
  <aside class="sidebar">
    <!-- Logo -->
    <div class="sidebar__logo">
      <div class="logo-icon">
        <IconBolt :size="18" />
      </div>
      <div class="logo-text">
        <span class="logo-title">WattTrack</span>
        <span class="logo-sub">RAG · Dutch Energy</span>
      </div>
    </div>

    <!-- New chat -->
    <button class="new-chat-btn" @click="$emit('new-chat')">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      Novo chat
    </button>

    <!-- Chat list -->
    <div class="sidebar__section-label">Conversas</div>
    <div class="sidebar__chats">
      <div v-if="loadingChats" class="chat-item-skeleton" v-for="i in 3" :key="i" />
      <template v-else>
        <button
          v-for="chat in chats"
          :key="chat.chat_id"
          class="chat-item"
          :class="{ active: chat.chat_id === activeChatId }"
          @click="$emit('select-chat', chat)"
        >
          <div class="chat-item__title">{{ chat.title }}</div>
          <div class="chat-item__meta">
            <span>{{ chat.message_count }} msg</span>
            <span class="dot" />
            <span>{{ formatDate(chat.updated_at) }}</span>
          </div>
        </button>
        <div v-if="!chats.length" class="sidebar__empty">
          Nenhum chat ainda
        </div>
      </template>
    </div>

    <!-- Model selector -->
    <div class="sidebar__bottom">
      <div class="sidebar__section-label">Modelo</div>
      <div class="model-select-wrap">
        <select class="model-select" :value="selectedModel" @change="$emit('change-model', $event.target.value)">
          <option value="qwen3.5-2b-unsloth">qwen3.5-2b-unsloth (padrão)</option>
          <option value="gemma4-unsloth">gemma4-unsloth</option>
        </select>
        <svg class="select-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
      </div>

      <!-- Status -->
    </div>
  </aside>
</template>

<script setup>
import IconBolt from './IconBolt.vue'

const props = defineProps({
  chats: { type: Array, default: () => [] },
  activeChatId: String,
  loadingChats: Boolean,
  online: Boolean,
  selectedModel: String,
})
defineEmits(['new-chat', 'select-chat', 'change-model'])

function formatDate(str) {
  if (!str) return ''
  const d = new Date(str)
  const now = new Date()
  const diff = now - d
  if (diff < 60000) return 'agora'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}min`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`
  return d.toLocaleDateString('pt-BR', { day: '2-digit', month: 'short' })
}
</script>

<style scoped>
.sidebar {
  width: 240px;
  min-width: 240px;
  background: var(--bg-panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  align-self: stretch;
}

.sidebar__logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
}

.logo-icon {
  width: 32px;
  height: 32px;
  background: var(--accent);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #0a0c0f;
  flex-shrink: 0;
}

.logo-text { display: flex; flex-direction: column; }
.logo-title { font-size: 15px; font-weight: 800; color: var(--text-primary); letter-spacing: -0.3px; }
.logo-sub { font-size: 10px; color: var(--text-secondary); font-family: var(--font-mono); letter-spacing: 0.05em; }

.new-chat-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 12px 4px;
  padding: 9px 14px;
  background: var(--accent);
  color: #0a0c0f;
  border: none;
  border-radius: var(--radius);
  font-size: 13px;
  font-weight: 700;
  transition: background 0.15s;
  font-family: var(--font-display);
}
.new-chat-btn:hover { background: var(--accent-dim); }

.sidebar__section-label {
  padding: 12px 16px 6px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.sidebar__chats {
  flex: 1;
  overflow-y: auto;
  padding: 0 8px;
}

.chat-item {
  display: block;
  width: 100%;
  padding: 9px 10px;
  margin-bottom: 2px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--radius);
  text-align: left;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
}
.chat-item:hover { background: var(--bg-hover); border-color: var(--border); }
.chat-item.active {
  background: var(--accent-glow);
  border-color: var(--accent);
}
.chat-item.active .chat-item__title { color: var(--accent); }

.chat-item__title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chat-item__meta {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 2px;
  font-family: var(--font-mono);
}
.dot { width: 3px; height: 3px; border-radius: 50%; background: var(--text-muted); }

.chat-item-skeleton {
  height: 52px;
  margin-bottom: 2px;
  border-radius: var(--radius);
  background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-hover) 50%, var(--bg-card) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
}

.sidebar__empty {
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
  padding: 24px 0;
  font-family: var(--font-mono);
}

.sidebar__bottom {
  border-top: 1px solid var(--border);
  padding: 8px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 111px;
  box-sizing: border-box;
}

.model-select-wrap {
  position: relative;
}

.model-select {
  width: 100%;
  padding: 8px 28px 8px 10px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-primary);
  font-size: 12px;
  appearance: none;
  cursor: pointer;
  font-family: var(--font-mono);
  transition: border-color 0.15s;
}
.model-select:focus { outline: none; border-color: var(--accent); }

.select-chevron {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary);
  pointer-events: none;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  font-size: 11px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.offline { background: var(--red); animation: pulse 2s infinite; }
.status-port { margin-left: auto; color: var(--text-muted); }
</style>
