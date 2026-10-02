export function between(text: string, open: string, close: string): string[] {
  const found: string[] = []
  let at = text.indexOf(open)
  while (at !== -1) {
    const end = text.indexOf(close, at + open.length)
    if (end === -1) break
    found.push(text.slice(at + open.length, end))
    at = text.indexOf(open, end + close.length)
  }
  return found
}

export function removeBetween(text: string, open: string, close: string): string {
  let out = ''
  let from = 0
  let at = text.indexOf(open)
  while (at !== -1) {
    const end = text.indexOf(close, at + open.length)
    if (end === -1) break
    out += text.slice(from, at)
    from = end + close.length
    at = text.indexOf(open, from)
  }
  return out + text.slice(from)
}

export const removeTags = (text: string) => removeBetween(text, '<', '>')

export function isOpen(text: string, open: string, close: string): boolean {
  return text.lastIndexOf(open) > text.lastIndexOf(close)
}

const isWordChar = (c: string) => c.toLowerCase() !== c.toUpperCase() || c === "'" || (c >= '0' && c <= '9')

export function words(text: string): string[] {
  const found: string[] = []
  let word = ''
  for (const c of text.toLowerCase()) {
    if (isWordChar(c)) word += c
    else if (word) {
      found.push(word)
      word = ''
    }
  }
  if (word) found.push(word)
  return found
}

export function containsPhrase(text: string, phrases: string[]): boolean {
  const padded = ` ${words(text).join(' ')} `
  return phrases.some((phrase) => padded.includes(` ${phrase} `))
}

export function sentences(text: string): string[] {
  const found: string[] = []
  let sentence = ''
  for (const [i, c] of [...text].entries()) {
    if (c === '\n') {
      found.push(sentence)
      sentence = ''
      continue
    }
    sentence += c
    if ('.!?'.includes(c) && (text[i + 1] === ' ' || text[i + 1] === '\n' || i === text.length - 1)) {
      found.push(sentence)
      sentence = ''
    }
  }
  found.push(sentence)
  return found.map((s) => s.trim()).filter(Boolean)
}

export function trimStartChars(text: string, chars: string): string {
  let i = 0
  while (i < text.length && chars.includes(text[i])) i++
  return text.slice(i)
}

export function count(text: string, part: string): number {
  return text.split(part).length - 1
}

export function tableCell(value: string | number): string {
  return String(value).replaceAll('|', '/').replaceAll('\n', ' ')
}

export function fileStamp(date = new Date()): string {
  return date.toISOString().slice(0, 16).replaceAll(':', '').replaceAll('T', '')
}
