export type Bucket =
  | 'TV=0' | 'TV<.05' | 'TV<.5' | 'TV<1' | 'TV=1'
  | 'val+' | 'val-' | 'shape!' | 'fail' | 'no-run';

export type AnswerShape =
  | 'value'
  | 'samples'
  | 'distribution'
  | { record: Record<string, AnswerShape> }
  | string;

export interface Atom {
  id: string;
  source: string;
  source_block_indices?: number[];
  task_type: string;
  eval_mode: string;
  answer_shape: AnswerShape;
  prompt: string;
  groundtruth_code: string;
  groundtruth_output: unknown;
  wrap_target?: string;
  notes?: string;
}

export interface ScoredRun {
  id: string;
  generation?: { code?: string };
  evaluation?: {
    gen?: { executed?: boolean; error?: string; answer?: unknown };
    comparison?: { ok?: boolean; error?: string };
    metrics?: Record<string, number>;
  };
}

export interface Collection {
  slug: string;
  label: string;
  jsonlPath: string;
  primaryRun?: string;
  description?: string;
}

export interface RunMeta {
  id: string;
  label: string;
  short: string;
  model: string;
  primer: boolean;
  thinking: boolean;
  primary: boolean;
}

/** Serialized output shape used by the chart + value/record renderers. */
export type RenderOutput =
  | { kind: 'value';        value: unknown }
  | { kind: 'distribution'; support: string[]; probs: number[] }
  | { kind: 'samples';      support: string[]; counts: number[] }
  | { kind: 'record';       fields: Record<string, RenderOutput> }
  | null;
