import { readFileSync } from 'node:fs';
import { dataPath } from '@voice/config';

export const LIBRARY_DIR = dataPath('benchmarks', 'longmemeval');
export const LIBRARY_SHEET = `${LIBRARY_DIR}/library-questions.md`;
export const LIBRARY_ROOM = 'longmemeval-library';

export interface LibraryQuestion {
  type: string;
  question: string;
  answer: string;
}

/** The questions written by load-library.ts; each is answerable from the loaded memory. */
export function libraryQuestions(): LibraryQuestion[] {
  return readFileSync(LIBRARY_SHEET, 'utf8')
    .split('\n')
    .filter((line) => line.startsWith('| ') && !line.startsWith('| type'))
    .map((line) => {
      const [type, , question, answer] = line.slice(2, -2).split(' | ');
      return { type: type.replace(' (abstention)', ''), question: question.trim(), answer: answer.trim() };
    });
}
