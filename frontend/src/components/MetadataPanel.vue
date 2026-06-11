<template>
  <div class="meta-panel" v-if="visible">
    <div class="meta-panel__header">
      <span>Painel de Métricas MLflow</span>
      <button class="close-btn" @click="$emit('close')">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>

    <div class="meta-panel__body" v-if="data">
      <div class="exp-badge">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14.5 10c-.83 0-1.5-.67-1.5-1.5v-5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5z"/><path d="M20.5 10H19V8.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/><path d="M9.5 14c.83 0 1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5S8 21.33 8 20.5v-5c0-.83.67-1.5 1.5-1.5z"/><path d="M3.5 14H5v1.5c0 .83-.67 1.5-1.5 1.5S2 16.33 2 15.5 2.67 14 3.5 14z"/><path d="M14 14.5c0-.83.67-1.5 1.5-1.5h5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5h-5c-.83 0-1.5-.67-1.5-1.5z"/><path d="M15.5 19H14v1.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5z"/><path d="M10 9.5C10 8.67 9.33 8 8.5 8h-5C2.67 8 2 8.67 2 9.5S2.67 11 3.5 11h5c.83 0 1.5-.67 1.5-1.5z"/><path d="M8.5 5H10V3.5C10 2.67 9.33 2 8.5 2S7 2.67 7 3.5 7.67 5 8.5 5z"/></svg>
        {{ data.experiment }}
      </div>
      <div class="runs-count">{{ data.total_runs }} run{{ data.total_runs !== 1 ? 's' : '' }}</div>

      <!-- Best run -->
      <div v-if="data.best_run" class="best-run">
        <div class="best-run__label">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          Melhor run
        </div>
        <div class="metrics-grid">
          <div class="metric-card" v-for="(val, key) in data.best_run.metrics" :key="key">
            <span class="metric-key">{{ key.toUpperCase() }}</span>
            <span class="metric-val">{{ typeof val === 'number' ? val.toFixed(3) : val }}</span>
          </div>
        </div>
        <div class="params-list" v-if="Object.keys(data.best_run.params).length">
          <div class="param-row" v-for="(val, key) in data.best_run.params" :key="key">
            <span class="param-key">{{ key }}</span>
            <span class="param-val">{{ val }}</span>
          </div>
        </div>
        <div class="run-status">
          <span class="status-pill" :class="data.best_run.status.toLowerCase()">{{ data.best_run.status }}</span>
          <span class="run-time">{{ formatTs(data.best_run.start_time) }}</span>
        </div>
      </div>

      <!-- All runs mini -->
      <div v-if="data.all_runs.length > 1" class="all-runs">
        <div class="all-runs__label">Todos os runs</div>
        <div class="run-row" v-for="run in data.all_runs" :key="run.run_id">
          <span class="run-row__dot" :class="run.status.toLowerCase()" />
          <span class="run-row__rmse" v-if="run.metrics.rmse">{{ run.metrics.rmse.toFixed(2) }}</span>
          <span class="run-row__time">{{ formatTs(run.start_time) }}</span>
        </div>
      </div>
    </div>

    <div class="meta-panel__loading" v-else-if="loading">
      <div class="spinner" />
      Carregando…
    </div>
    <div class="meta-panel__error" v-else-if="error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { api } from '../api.js'

const props = defineProps({ visible: Boolean })
defineEmits(['close'])

const data = ref(null)
const loading = ref(false)
const error = ref(null)

watch(() => props.visible, async (v) => {
  if (!v || data.value) return
  loading.value = true
  error.value = null
  try {
    data.value = await api.metadata()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}, { immediate: true })

function formatTs(str) {
  if (!str) return '—'
  const d = new Date(str)
  return d.toLocaleString('pt-BR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}
</script>

<style scoped>
.meta-panel {
  width: 260px;
  min-width: 260px;
  background: var(--bg-panel);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  align-self: stretch;
  overflow: hidden;
  animation: slideIn 0.2s ease both;
}

.meta-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  font-weight: 700;
  font-family: var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--accent);
}

.close-btn {
  background: none;
  border: none;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  padding: 3px;
  border-radius: 4px;
  transition: color 0.15s, background 0.15s;
}
.close-btn:hover { color: var(--text-primary); background: var(--bg-hover); }

.meta-panel__body {
  flex: 1;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.exp-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 5px 8px;
}

.runs-count {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-muted);
}

.best-run {
  background: var(--bg-card);
  border: 1px solid var(--border-bright);
  border-radius: var(--radius);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.best-run__label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
  font-family: var(--font-mono);
}

.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.metric-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 7px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.metric-key { font-size: 9px; font-family: var(--font-mono); color: var(--text-muted); letter-spacing: 0.1em; }
.metric-val { font-size: 16px; font-weight: 700; color: var(--accent); font-family: var(--font-mono); }

.params-list { display: flex; flex-direction: column; gap: 4px; }
.param-row {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 11px;
  font-family: var(--font-mono);
}
.param-key { color: var(--text-secondary); flex-shrink: 0; }
.param-val { color: var(--text-primary); margin-left: auto; text-align: right; }

.run-status { display: flex; align-items: center; gap: 8px; }
.status-pill {
  font-size: 9px;
  font-family: var(--font-mono);
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 20px;
  border: 1px solid;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.status-pill.finished { border-color: var(--green); color: var(--green); background: var(--green-dim); }
.status-pill.failed, .status-pill.error { border-color: var(--red); color: var(--red); background: var(--red-dim); }
.status-pill.running { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }

.run-time { font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); margin-left: auto; }

.all-runs { display: flex; flex-direction: column; gap: 4px; }
.all-runs__label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  font-family: var(--font-mono);
  margin-bottom: 2px;
}
.run-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  font-family: var(--font-mono);
  padding: 4px 6px;
  border-radius: 4px;
  background: var(--bg-card);
}
.run-row__dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.run-row__dot.finished { background: var(--green); }
.run-row__dot.failed { background: var(--red); }
.run-row__dot.running { background: var(--accent); }
.run-row__rmse { color: var(--accent); font-weight: 700; }
.run-row__time { color: var(--text-muted); margin-left: auto; font-size: 10px; }

.meta-panel__loading, .meta-panel__error {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
}
.meta-panel__error { color: var(--red); }

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
</style>
