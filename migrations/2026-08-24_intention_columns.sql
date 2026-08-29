-- Supabase focus_logs: the intention columns the local sink already writes.
-- SupabaseSink._flatten_row now sends both (kept in lockstep with
-- monitor/local_sink.py), so this must be applied before deploying that code
-- or every mirrored insert fails on the unknown columns.
alter table public.focus_logs add column if not exists intention_id integer;
alter table public.focus_logs add column if not exists align_score real;
