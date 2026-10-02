import { llm } from '@livekit/agents'

export const instructions = (now: Date) => `You are a helpful voice assistant running fully on local models.
You hear the user's voice directly; the text of their earlier turns is a transcript.
Keep replies short and conversational: one or two sentences, no markdown, lists or emoji.
Use tools only when the user asks for that action. After a tool runs, tell the user what it did.
You know nothing about the user's past except what the recall tool returns. Whenever the user asks
about their own life, past, plans, purchases, people they know, or anything they told you before, call
recall first, every time, before answering. Answer from what it returns and say when it was; if the user
said different things at different times, the most recent is current. If recall doesn't answer the
question, say you do not remember. Never answer questions about the user from your own knowledge.
If the user corrects a reminder or says undo, cancel it (and set the corrected one).
You have no weather, news or internet tools: say so instead of guessing.
The current local time is ${now.toTimeString().slice(0, 5)}, ${now.toDateString()}.`

export const RECALL_EXAMPLES = [
  {
    user: "What did I say my sister's new job was?",
    query: "What did I say my sister's new job was?",
    result: '[Mon 3 Aug, 18:20] User: My sister Anna just started as a nurse in Lisbon.',
    reply: 'You told me on August 3rd that Anna started as a nurse in Lisbon.',
  },
  {
    user: 'Can you recommend a podcast for my commute?',
    query: 'podcasts or topics the user likes',
    result: '[Tue 11 Aug, 08:05] User: I love history and true crime, I just finished Hardcore History.',
    reply: 'Since you loved Hardcore History, try The Rest Is History: same depth, shorter episodes.',
  },
]

export function exampleChatCtx(): llm.ChatContext {
  const ctx = llm.ChatContext.empty()
  RECALL_EXAMPLES.forEach((e, i) => {
    const callId = `example_${i}`
    ctx.addMessage({ role: 'user', content: e.user })
    ctx.insert(llm.FunctionCall.create({ callId, name: 'recall', args: JSON.stringify({ question: e.query }) }))
    ctx.insert(llm.FunctionCallOutput.create({ callId, name: 'recall', output: e.result, isError: false }))
    ctx.addMessage({ role: 'assistant', content: e.reply })
  })
  return ctx
}

export function exampleMessages() {
  return RECALL_EXAMPLES.flatMap((e, i) => [
    { role: 'user', content: e.user },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: `example_${i}`,
          type: 'function',
          function: { name: 'recall', arguments: JSON.stringify({ question: e.query }) },
        },
      ],
    },
    { role: 'tool', tool_call_id: `example_${i}`, content: e.result },
    { role: 'assistant', content: e.reply },
  ])
}
