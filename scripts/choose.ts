import { checkbox, select } from '@inquirer/prompts'

const AGENTS = [
  { value: 'cascade', name: 'cascade', description: 'Parakeet → Gemma 4 26B (LM Studio) → Kokoro. The default.' },
  { value: 's2s', name: 's2s', description: 'NVIDIA VoiceChat 11B, full-duplex speech-to-speech.' },
  { value: 'omni', name: 'omni', description: 'Qwen3-Omni 30B hears your audio, Kokoro speaks. ~24 GB.' },
]

const context = { output: process.stderr }

try {
  if (process.argv[2] === 'agent') {
    const agent = await select({ message: 'Which agent?', choices: AGENTS, default: 'cascade' }, context)
    console.log(agent)
  } else if (process.argv[2] === 'models') {
    const choices = AGENTS.map((a) => (a.value === 'cascade' ? { ...a, checked: true, disabled: '(always)' } : a))
    const extra = await checkbox({ message: 'Fetch models for which agents?', choices }, context)
    console.log(extra.join(' '))
  } else {
    throw new Error('usage: choose.ts agent|models')
  }
} catch (err) {
  if ((err as Error).name === 'ExitPromptError') process.exit(130)
  throw err
}
