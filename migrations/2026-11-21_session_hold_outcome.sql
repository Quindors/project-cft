-- Sessions on hold, and the complete/incomplete answer (monitor/intentions.py).
--
-- intentions gains three columns the monitor now mirrors:
--   paused_sec  seconds the session stood still (pauses, and the time the
--               dashboard was the window in front), so its length is its
--               window minus this
--   held_at     when a held session was shelved; NULL otherwise. A held
--               session has status 'held': neither active nor finished
--   outcome     the user's answer to "did you finish it?" for a finished
--               session: complete | incomplete | skipped, NULL if unanswered
--
-- Safe to apply before or after the exe built from this commit: until it is
-- applied the monitor's mirror falls back to the columns that already exist
-- (supa_mirror._upsert_intentions), so sessions keep arriving and only the
-- pause time and the outcome are missing from the hosted dashboard's offline
-- forest.

alter table public.intentions add column if not exists paused_sec integer not null default 0;
alter table public.intentions add column if not exists held_at text;
alter table public.intentions add column if not exists outcome text;

-- sessions_v gains paused_sec and outcome at the end, the same shape as the
-- monitor's GET /api/sessions. Appending columns is what `create or replace`
-- allows, so nothing is dropped and the grant stays.
create or replace view public.sessions_v
with (security_invoker = true) as
select i.id, i.text, i.started_at, i.ended_at, i.status, i.llm_category,
       i.project_id,
       p.name as project_name,
       (select avg(coalesce(case f.human_align when 1 then 1.0 when 0 then 0.0 end,
                            f.align_score))
          from public.focus_logs f where f.intention_id = i.id) as avg_align,
       t.cat   as top_category,
       t.share as top_share,
       i.paused_sec,
       i.outcome
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
