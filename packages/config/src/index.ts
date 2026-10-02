import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const repoRoot = fileURLToPath(new URL('../../../', import.meta.url))

export function env(name: string): string {
  const value = process.env[name]
  if (!value) throw new Error(`${name} is not set: run \`npm run setup\` or copy .env.example to .env`)
  return value
}

export function dataPath(...parts: string[]): string {
  return join(repoRoot, 'data', ...parts)
}
