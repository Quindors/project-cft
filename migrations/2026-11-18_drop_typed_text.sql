-- 2026-11-18: the monitor no longer reads the keyboard.
--
-- Builds before this date ran a pynput keyboard hook and wrote the text typed
-- between decisions into focus_logs.typed_text, mirrored here from the local
-- database. Nothing ever read the column (the classifier prompt never carried
-- it, the dashboard never showed it). The hook and the column are gone from
-- the app; this removes the column, and everything in it, from the cloud copy.
--
-- Run AFTER the new build is live: an older exe still mirrors typed_text and
-- PostgREST rejects inserts naming a column that does not exist.

alter table public.focus_logs drop column if exists typed_text;
