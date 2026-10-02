export function option(name: string, fallback: string): string {
  const at = process.argv.indexOf(name)
  return at > 0 && at + 1 < process.argv.length ? process.argv[at + 1] : fallback
}

export const flag = (name: string) => process.argv.includes(name)

export function percentile(values: (number | undefined)[], p: number): number {
  const sorted = values.filter((v): v is number => v !== undefined && Number.isFinite(v)).sort((a, b) => a - b)
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : Number.NaN
}

export const median = (values: (number | undefined)[]) => percentile(values, 0.5)

export const percent = (count: number, of: number) => (of ? `${Math.round((100 * count) / of)}%` : '-')
