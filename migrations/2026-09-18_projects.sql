-- Projects: the long-lived thing a focus session is part of (monitor/projects.py).
-- Mirror of the monitor's `projects` table plus the new intentions.project_id
-- column, so the hosted dashboard can name and filter the forest's trees by
-- project while the monitor is off. Local SQLite stays the source of truth:
-- the exe upserts every change and backfills all rows at startup.
--
-- Apply BEFORE running the exe built from this commit: the intention upserts
-- carry project_id from then on and fail until the column exists.

create table if not exists public.projects (
  id          integer primary key,
  name        text not null,
  description text not null default '',
  category    text,
  status      text not null default 'active',
  created_at  text not null,
  archived_at text
);

alter table public.projects enable row level security;
-- Same trust model as intentions (see 2026-09-03_intentions_mirror.sql). A
-- project row is a label, not history, so unlike intentions the exe may
-- delete one: the monitor unlinks the sessions first, then removes the row.
drop policy if exists projects_select on public.projects;
drop policy if exists projects_insert on public.projects;
drop policy if exists projects_update on public.projects;
drop policy if exists projects_delete on public.projects;
create policy projects_select on public.projects for select to anon, authenticated using (true);
create policy projects_insert on public.projects for insert to anon, authenticated with check (true);
create policy projects_update on public.projects for update to anon, authenticated using (true) with check (true);
create policy projects_delete on public.projects for delete to anon, authenticated using (true);
revoke truncate on table public.projects from anon, authenticated;

alter table public.intentions add column if not exists project_id integer;

-- sessions_v gains the project (id and name) — same shape as the monitor's
-- GET /api/sessions, so the forest reads either source identically. Dropped
-- first: `create or replace` cannot insert columns ahead of existing ones.
drop view if exists public.sessions_v;
create view public.sessions_v
with (security_invoker = true) as
select i.id, i.text, i.started_at, i.ended_at, i.status, i.llm_category,
       i.project_id,
       p.name as project_name,
       (select avg(f.align_score) from public.focus_logs f where f.intention_id = i.id) as avg_align,
       t.cat   as top_category,
       t.share as top_share
from public.intentions i
left join public.projects p on p.id = i.project_id
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
