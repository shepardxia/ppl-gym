-- Rename atom_id -> problem_id (the dataset is problem-centric since the P2
-- redesign; existing rows already hold problem_ids in this column).

ALTER TABLE feedback RENAME COLUMN atom_id TO problem_id;

DROP INDEX IF EXISTS feedback_atom_id_idx;
CREATE INDEX IF NOT EXISTS feedback_problem_id_idx ON feedback (problem_id, created_at DESC);
