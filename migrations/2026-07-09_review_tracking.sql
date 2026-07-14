-- Learning-loop schema: review tracking + accuracy/confusion views.
--
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
-- It is additive and non-destructive: no existing rows are modified.
--
-- Why: today a focus_logs row only gets a human_label when the AI was WRONG,
-- so the labeled set is 100% corrections and true accuracy is undefined. The
-- review workflow (review.py) sets human_label on every reviewed row -- equal
-- to category_name when the AI was right ("confirmed"), different when it was
-- wrong ("correction") -- and stamps reviewed_at. These views then read that.

ALTER TABLE public.focus_logs
  ADD COLUMN IF NOT EXISTS reviewed_at timestamptz;

COMMENT ON COLUMN public.focus_logs.reviewed_at IS
  'When a human reviewed this classification (via review.py or the dashboard). '
  'NULL = not yet reviewed. On review, human_label is always set: equal to '
  'category_name means confirmed-correct, different means a correction.';

-- Daily accuracy over reviewed rows. A row counts as reviewed once it has a
-- non-empty human_label (this also covers legacy corrections made before
-- reviewed_at existed). Correct = human_label matches the AI's category_name.
CREATE OR REPLACE VIEW public.v_accuracy_daily AS
SELECT
  (timestamp AT TIME ZONE 'UTC')::date AS day,
  count(*) FILTER (
    WHERE human_label IS NOT NULL AND human_label <> ''
  ) AS reviewed,
  count(*) FILTER (
    WHERE human_label IS NOT NULL AND human_label <> ''
      AND lower(human_label) = lower(coalesce(category_name, ''))
  ) AS confirmed,
  round(
    100.0 * count(*) FILTER (
      WHERE human_label IS NOT NULL AND human_label <> ''
        AND lower(human_label) = lower(coalesce(category_name, ''))
    )
    / nullif(count(*) FILTER (WHERE human_label IS NOT NULL AND human_label <> ''), 0)
  , 1) AS accuracy_pct
FROM public.focus_logs
GROUP BY 1
ORDER BY 1;

-- Confusion pairs: what the AI said (ai_category) vs what the human corrected
-- it to (corrected_category), with a count. Highest-count rows are the most
-- systematic mistakes -- the ones tune_prompt.py targets.
CREATE OR REPLACE VIEW public.v_confusion AS
SELECT
  coalesce(category_name, '(none)') AS ai_category,
  human_label                       AS corrected_category,
  count(*)                          AS n
FROM public.focus_logs
WHERE human_label IS NOT NULL
  AND human_label <> ''
  AND lower(human_label) <> lower(coalesce(category_name, ''))
GROUP BY 1, 2
ORDER BY n DESC;
