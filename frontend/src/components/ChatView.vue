<template>
  <div class="chat-view">

    <!-- Header -->
    <div class="chat-header">
      <div class="chat-header__title">
        <span v-if="activeChatId">{{ activeTitle || 'Chat' }}</span>
        <span v-else class="chat-header__placeholder">Selecione ou crie um chat</span>
      </div>
      <div class="chat-header__actions">
        <button
          class="header-btn"
          :class="{ active: showMeta }"
          @click="$emit('toggle-meta')"
          title="Painel MLflow"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          Métricas
        </button>
      </div>
    </div>

    <!-- Messages area -->
    <div class="messages-area" ref="scrollEl">
      <!-- Welcome screen -->
      <div v-if="!activeChatId" class="welcome">
        <div class="welcome__icon">
          <IconBolt :size="28" />
        </div>
        <h2 class="welcome__title">Dutch Energy RAG</h2>
        <p class="welcome__desc">
          Consulte dados de consumo elétrico residencial.<br>
        </p>
      </div>

      <!-- Messages (single container avoids unmounting the list mid-response) -->
      <div v-else class="messages-list">
        <div v-if="messages.length === 0 && !loading" class="empty-chat">
          <div class="empty-chat__icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          </div>
          <p>Comece a conversa enviando uma mensagem.</p>
        </div>

        <ChatMessage
          v-for="msg in messages"
          :key="msg.id"
          :msg="msg"
          :delay="0"
        />
        <TypingIndicator v-if="loading" />
      </div>
    </div>

    <!-- Input area -->
    <div class="input-area" :class="{ disabled: !activeChatId && false }">
      <div class="input-wrap">
        <textarea
          ref="textareaEl"
          v-model="draft"
          class="input-textarea"
          placeholder="Pergunte sobre o pipeline, métricas, modelo..."
          rows="1"
          :disabled="loading"
          @keydown.enter.exact.prevent="submit"
          @keydown.enter.shift.exact="newline"
          @input="autoResize"
        />
        <button
          class="send-btn"
          :disabled="!draft.trim() || loading"
          @click="submit"
        >
          <svg v-if="!loading" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          <div v-else class="spinner-sm" />
        </button>
      </div>
      <div class="input-hint">
        <span>Enter para enviar · Shift+Enter para nova linha</span>
        <span v-if="error" class="input-error">{{ error }}</span>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import ChatMessage from './ChatMessage.vue'
import TypingIndicator from './TypingIndicator.vue'
import IconBolt from './IconBolt.vue'

const props = defineProps({
  activeChatId: String,
  activeTitle: String,
  messages: { type: Array, default: () => [] },
  loading: Boolean,
  error: String,
  showMeta: Boolean,
})

const emit = defineEmits(['send', 'toggle-meta'])

const draft = ref('')
const scrollEl = ref(null)
const textareaEl = ref(null)

function submit() {
  const msg = draft.value.trim()
  if (!msg || props.loading) return
  emit('send', msg)
  draft.value = ''
  nextTick(() => {
    if (textareaEl.value) {
      textareaEl.value.style.height = 'auto'
    }
  })
}

function newline() {
  draft.value += '\n'
  nextTick(autoResize)
}

function autoResize() {
  nextTick(() => {
    const el = textareaEl.value
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  })
}

watch(
  () => [props.messages.length, props.loading],
  async () => {
    await nextTick()
    if (scrollEl.value) {
      scrollEl.value.scrollTop = scrollEl.value.scrollHeight
    }
  },
)
</script>

<style scoped>
.chat-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 52px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
  flex-shrink: 0;
}

.chat-header__title {
  font-size: 15px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chat-header__placeholder { color: var(--text-muted); font-weight: 400; }

.chat-header__actions { display: flex; gap: 8px; }

.header-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text-secondary);
  font-size: 12px;
  font-family: var(--font-display);
  font-weight: 600;
  transition: all 0.15s;
}
.header-btn:hover { border-color: var(--accent); color: var(--accent); }
.header-btn.active { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }

/* Messages */
.messages-area {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 24px 8px;
  background: var(--bg);
  display: flex;
  flex-direction: column;
}

.welcome, .empty-chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  text-align: center;
  max-width: 480px;
  margin: 0 auto;
  width: 100%;
}

.welcome__icon {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  background: var(--accent);
  color: #0a0c0f;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
}

.welcome__title {
  font-size: 24px;
  font-weight: 800;
  letter-spacing: -0.5px;
}

.welcome__desc {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.65;
}

.suggestion-chip {
  padding: 7px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 13px;
  color: var(--text-secondary);
  font-family: var(--font-display);
  transition: all 0.15s;
}
.suggestion-chip:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-glow);
}

.empty-chat__icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
}
.empty-chat p { color: var(--text-secondary); font-size: 13px; }

.messages-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding-bottom: 8px;
  flex: 1;
}

.messages-list .empty-chat {
  flex: 1;
  min-height: 200px;
}

/* Input */
.input-area {
  border-top: 1px solid var(--border);
  background: var(--bg-panel);
  padding: 14px 20px 14px;
  flex-shrink: 0;
}

.input-wrap {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 10px 10px 10px 14px;
  transition: border-color 0.15s;
}
.input-wrap:focus-within { border-color: var(--accent); }

.input-textarea {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-size: 14px;
  resize: none;
  line-height: 1.6;
  min-height: 24px;
  max-height: 160px;
  overflow-y: auto;
}
.input-textarea::placeholder { color: var(--text-muted); }
.input-textarea:disabled { opacity: 0.5; }

.send-btn {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: var(--accent);
  border: none;
  color: #0a0c0f;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: background 0.15s, opacity 0.15s;
}
.send-btn:hover:not(:disabled) { background: var(--accent-dim); }
.send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

.spinner-sm {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(0,0,0,0.2);
  border-top-color: #0a0c0f;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

.input-hint {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
  font-size: 11px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  padding: 0 2px;
}

.input-error {
  color: var(--red);
  font-size: 11px;
}
</style>
