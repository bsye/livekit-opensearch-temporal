import { containsPhrase } from '@voice/text'

const FIRST_PERSON = ['i', 'my', 'me', 'mine', "i'm", "i've", "i'd", "i'll"]
const REFERS_BACK = [
  'previous',
  'earlier',
  'last time',
  'before',
  'remind me what',
  ...['talked', 'discussed', 'spoke', 'said'].map((verb) => `we ${verb}`),
  ...['told', 'said', 'mentioned', 'recommended', 'suggested'].map((verb) => `you ${verb}`),
  ...['tell', 'mention', 'say'].map((verb) => `did i ${verb}`),
]

export const isFirstPerson = (text: string) => containsPhrase(text, FIRST_PERSON)
export const refersBack = (text: string) => containsPhrase(text, REFERS_BACK)
