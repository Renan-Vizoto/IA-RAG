<template>
  <div class="typing-wrap">
    <div class="typing-avatar">
      <IconBolt :size="13" />
    </div>
    <div class="typing-bubble">
      <span class="dot" style="animation-delay: 0ms" />
      <span class="dot" style="animation-delay: 160ms" />
      <span class="dot" style="animation-delay: 320ms" />
    </div>
  </div>
</template>

<script setup>
import IconBolt from './IconBolt.vue'
</script>

<style scoped>
.typing-wrap {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  animation: fadeUp 0.2s ease both;
}

.typing-avatar {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: var(--accent);
  color: #0a0c0f;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 2px;
}

.typing-bubble {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 4px 12px 12px 12px;
  padding: 14px 18px;
  display: flex;
  gap: 5px;
  align-items: center;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.1s ease-in-out infinite;
  display: inline-block;
}
</style>
