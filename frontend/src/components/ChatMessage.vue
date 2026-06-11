<template>
  <div class="msg" :class="msg.role" :style="{ animationDelay: delay + 'ms' }">

    <!-- Avatar -->
    <div class="msg__avatar">
      <template v-if="msg.role === 'assistant'">
        <IconBolt :size="13" />
      </template>
      <template v-else>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>
      </template>
    </div>

    <!-- Content -->
    <div class="msg__body">
      <div class="msg__role">{{ msg.role === 'assistant' ? 'WattTrack' : 'Você' }}</div>
      <div class="msg__text" v-html="renderText(msg.content)" />

      <!-- Search results (assistant only) -->
      <div v-if="msg.search_results && msg.search_results.length" class="msg__sources">
        <button class="sources-toggle" @click="showSources = !showSources">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          {{ msg.search_results.length }} fonte{{ msg.search_results.length > 1 ? 's' : '' }} recuperada{{ msg.search_results.length > 1 ? 's' : '' }}
          <svg class="chevron" :class="{ open: showSources }" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
        </button>
        <transition name="sources">
          <div v-if="showSources" class="sources-list">
            <div v-for="(hit, i) in msg.search_results" :key="i" class="source-item">
              <div class="source-item__header">
                <span class="source-badge">{{ hit.source || 'governance' }}</span>
                <span class="source-dist">d={{ (hit.distance || 0).toFixed(3) }}</span>
              </div>
              <p class="source-item__text">{{ truncate(hit.text, 160) }}</p>
            </div>
          </div>
        </transition>
      </div>

      <!-- Metadata (confidence, time, model) -->
      <div v-if="msg.role === 'assistant' && formatTime(msg.response_time_seconds)" class="msg__meta">
        <span v-if="msg.confidence_score != null">
          <span class="meta-label">conf</span>
          <span :class="confClass(msg.confidence_score)">{{ (Number(msg.confidence_score) * 100).toFixed(0) }}%</span>
        </span>
        <span class="meta-sep" />
        <span><span class="meta-label">tempo</span> {{ formatTime(msg.response_time_seconds) }}</span>
        <span class="meta-sep" />
        <span><span class="meta-label">modelo</span> {{ msg.model || '—' }}</span>
        <span v-if="msg.tokens" class="meta-sep" />
        <span v-if="msg.tokens"><span class="meta-label">tokens</span> {{ msg.tokens.total_tokens ?? '—' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import IconBolt from './IconBolt.vue'

const props = defineProps({
  msg: Object,
  delay: { type: Number, default: 0 },
})

const showSources = ref(false)

function formatTime(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value)) return null
  return `${value.toFixed(1)}s`
}

function renderText(text) {
  if (text == null) return ''
  const safe = String(text)
  return safe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
}

function truncate(str, n) {
  if (!str) return ''
  return str.length > n ? str.slice(0, n) + '…' : str
}

function confClass(score) {
  if (score >= 0.7) return 'conf-high'
  if (score >= 0.4) return 'conf-mid'
  return 'conf-low'
}
</script>

<style scoped>
.msg {
  display: flex;
  gap: 12px;
  padding: 4px 0;
  animation: fadeUp 0.25s ease both;
}

.msg.user { flex-direction: row-reverse; }
.msg.user .msg__body { align-items: flex-end; }
.msg.user .msg__role { text-align: right; }
.msg.user .msg__text { background: var(--bg-card); border-color: var(--border-bright); }

.msg__avatar {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}
.msg.assistant .msg__avatar { background: var(--accent); color: #0a0c0f; }
.msg.user .msg__avatar { background: var(--bg-card); border: 1px solid var(--border-bright); color: var(--text-secondary); }

.msg__body {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-width: min(560px, 78%);
}

.msg__role {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  font-family: var(--font-mono);
}
.msg.assistant .msg__role { color: var(--accent); }

.msg__text {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 4px 12px 12px 12px;
  padding: 10px 14px;
  font-size: 14px;
  line-height: 1.65;
  color: var(--text-primary);
  word-break: break-word;
}
.msg.user .msg__text { border-radius: 12px 4px 12px 12px; }

.msg__text :deep(strong) { color: var(--accent); font-weight: 700; }
.msg__text :deep(code) {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg);
  border: 1px solid var(--border-bright);
  padding: 1px 5px;
  border-radius: 3px;
  color: var(--green);
}

/* Sources */
.msg__sources { margin-top: 4px; }

.sources-toggle {
  display: flex;
  align-items: center;
  gap: 5px;
  background: none;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 10px;
  font-size: 11px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  transition: border-color 0.15s, color 0.15s;
}
.sources-toggle:hover { border-color: var(--accent); color: var(--accent); }

.chevron { transition: transform 0.2s; }
.chevron.open { transform: rotate(180deg); }

.sources-list { margin-top: 6px; display: flex; flex-direction: column; gap: 6px; }

.source-item {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 10px;
}

.source-item__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.source-badge {
  font-size: 10px;
  font-family: var(--font-mono);
  font-weight: 700;
  padding: 1px 6px;
  background: var(--accent-glow);
  color: var(--accent);
  border-radius: 3px;
  border: 1px solid rgba(232,200,74,0.25);
}

.source-dist {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  margin-left: auto;
}

.source-item__text {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.sources-enter-active { transition: all 0.2s ease; }
.sources-leave-active { transition: all 0.15s ease; }
.sources-enter-from { opacity: 0; transform: translateY(-4px); }
.sources-leave-to { opacity: 0; }

/* Metadata row */
.msg__meta {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  margin-top: 2px;
}

.meta-label { opacity: 0.6; margin-right: 2px; }
.meta-sep { width: 3px; height: 3px; border-radius: 50%; background: var(--border-bright); }

.conf-high { color: var(--green); }
.conf-mid  { color: var(--accent); }
.conf-low  { color: var(--red); }
</style>
