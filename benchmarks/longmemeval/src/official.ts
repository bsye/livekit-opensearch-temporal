// Line-for-line ports of LongMemEval's official retrieval evaluation, so results are comparable
// with published numbers:
//   src/retrieval/eval_utils.py            → dcg, ndcg, evaluateRetrieval, evaluateRetrievalTurn2Session
//   src/retrieval/run_retrieval.py         → processItemFlatIndex (corpus + labels), K values,
//                                            and the abstention / no-target exclusions
// https://github.com/xiaowu0162/LongMemEval (MIT)

export const KS = [1, 3, 5, 10, 30, 50];
export type Granularity = 'session' | 'turn';

export interface Turn {
  role: 'user' | 'assistant';
  content: string;
  has_answer?: boolean;
}

export interface Question {
  question_id: string;
  question_type: string;
  question: string;
  question_date: string;
  answer: string;
  haystack_session_ids: string[];
  haystack_dates: string[];
  haystack_sessions: Turn[][];
  answer_session_ids: string[];
}

export interface Doc {
  id: string; // corpus id; contains "answer" iff it is evidence
  text: string;
  timestamp: string;
}

/** process_item_flat_index: user content only; evidence ids keep "answer", others get "noans". */
export function processItemFlatIndex(session: Turn[], granularity: Granularity, sessId: string, timestamp: string): Doc[] {
  if (granularity === 'session') {
    const text = session.filter((t) => t.role === 'user').map((t) => t.content).join(' ');
    let id = sessId;
    if (sessId.includes('answer') && session.filter((t) => t.role === 'user').every((t) => !t.has_answer)) {
      id = sessId.replace('answer', 'noans');
    }
    return [{ id, text, timestamp }];
  }
  const docs: Doc[] = [];
  session.forEach((turn, i) => {
    if (turn.role !== 'user') return;
    let id = `${sessId}_${i + 1}`;
    if (sessId.includes('answer') && !turn.has_answer) id = id.replace('answer', 'noans');
    docs.push({ id, text: turn.content, timestamp });
  });
  return docs;
}

export function buildCorpus(q: Question, granularity: Granularity): Doc[] {
  return q.haystack_session_ids.flatMap((sid, i) =>
    processItemFlatIndex(q.haystack_sessions[i], granularity, sid, q.haystack_dates[i]),
  );
}

/** Questions the official script leaves out of the averages. */
export function excluded(q: Question): 'abstention' | 'no-target' | undefined {
  if (q.question_id.includes('_abs')) return 'abstention';
  const userTurns = q.haystack_sessions.flat().filter((t) => t.role === 'user');
  if (!userTurns.some((t) => t.has_answer)) return 'no-target';
  return undefined;
}

function dcg(relevances: number[], k: number): number {
  const r = relevances.slice(0, k);
  if (r.length === 0) return 0;
  let sum = r[0];
  for (let i = 1; i < r.length; i++) sum += r[i] / Math.log2(i + 1);
  return sum;
}

function ndcg(rankings: number[], correctDocs: Set<string>, corpusIds: string[], k: number): number {
  const relevances = corpusIds.map((id) => (correctDocs.has(id) ? 1 : 0));
  const sorted = rankings.slice(0, k).map((idx) => relevances[idx]);
  const ideal = dcg([...relevances].sort((a, b) => b - a), k);
  return ideal === 0 ? 0 : dcg(sorted, k) / ideal;
}

export interface Scores {
  recall_any: number;
  recall_all: number;
  ndcg_any: number;
}

export function evaluateRetrieval(rankings: number[], correctDocs: string[], corpusIds: string[], k: number): Scores {
  const recalled = new Set(rankings.slice(0, k).map((idx) => corpusIds[idx]));
  const correct = new Set(correctDocs);
  return {
    recall_any: correctDocs.some((d) => recalled.has(d)) ? 1 : 0,
    recall_all: correctDocs.every((d) => recalled.has(d)) ? 1 : 0,
    ndcg_any: ndcg(rankings, correct, corpusIds, k),
  };
}

export function evaluateRetrievalTurn2Session(rankings: number[], correctDocs: string[], corpusIds: string[], k: number): Scores {
  const strip = (id: string) => id.split('_').slice(0, -1).join('_');
  const correct = [...new Set(correctDocs.map(strip))];
  const ids = corpusIds.map(strip);
  let effectiveK = k;
  let unique = new Set(rankings.slice(0, effectiveK).map((idx) => ids[idx]));
  while (effectiveK <= ids.length && unique.size < k) {
    effectiveK += 1;
    unique = new Set(rankings.slice(0, effectiveK).map((idx) => ids[idx]));
  }
  return evaluateRetrieval(rankings, correct, ids, effectiveK);
}
