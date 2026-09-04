-- Mirror of the monitor's `intentions` table (session intentions), so the hosted
-- dashboard can draw the forest and the session history while the monitor is
-- off. Local SQLite stays the source of truth: the exe upserts every change
-- (monitor/supa_mirror.mirror_intention) and backfills all rows at startup.
--
-- Timestamps are the monitor's naive local "YYYY-MM-DD HH:MM:SS" strings, kept
-- as text on purpose: the dashboard parses them exactly the way it parses
-- focus_logs, and the monitor's own clock is the only one that matters here.
--
-- Apply BEFORE running the exe built from this commit: the mirror upserts
-- start at launch and fail (harmlessly, but noisily) until the table exists.

create table if not exists public.intentions (
  id           integer primary key,
  text         text not null,
  started_at   text not null,
  ends_at      text not null,
  ended_at     text,
  status       text not null,
  llm_category text
);

alter table public.intentions enable row level security;
-- Same trust model as focus_logs (see 2026-08-28_enable_rls.sql): the anon key
-- is the exe, the dashboard and the phone. The exe needs insert + update (an
-- upsert); readers need select; nothing needs delete.
create policy intentions_select on public.intentions for select to anon, authenticated using (true);
create policy intentions_insert on public.intentions for insert to anon, authenticated with check (true);
create policy intentions_update on public.intentions for update to anon, authenticated using (true) with check (true);
revoke delete, truncate on table public.intentions from anon, authenticated;

-- Finished sessions in the same shape as the monitor's GET /api/sessions, so
-- the dashboard's forest reads either source identically: mean ALIGN over the
-- session's scored rows, and the most-logged effective category (a human
-- correction wins; Ambiguous/System/error rows never count) with its share.
create or replace view public.sessions_v
with (security_invoker = true) as
select i.id, i.text, i.started_at, i.ended_at, i.status, i.llm_category,
       (select avg(f.align_score) from public.focus_logs f where f.intention_id = i.id) as avg_align,
       t.cat   as top_category,
       t.share as top_share
from public.intentions i
left join lateral (
  select cat, round((n::numeric / sum(n) over ()), 3)::float8 as share
  from (
    select coalesce(nullif(trim(f.human_label), ''), f.category_name) as cat, count(*) as n
    from public.focus_logs f
    where f.intention_id = i.id
    group by 1
  ) s
  where cat is not null
    and lower(cat) not in ('ambiguous', 'system', '[error]', '[parse_error]')
  order by n desc
  limit 1
) t on true
where i.status in ('completed', 'expired');

grant select on public.sessions_v to anon, authenticated;
