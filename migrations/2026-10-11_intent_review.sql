-- Intent rulings from the Review tab (monitor/intent_review.py).
-- focus_logs.human_align: 1 = the user ruled the row on-intent, 0 = off-intent,
-- NULL = no ruling. The model's align_score is kept untouched; readers take the
-- ruling over it, exactly as human_label is taken over category_name.
--
-- Until this is applied, the exe's mirror of a ruling fails harmlessly in the
-- background and the hosted dashboard's offline forest uses the model's score.

alter table public.focus_logs add column if not exists human_align integer;

-- sessions_v: the forest's mean alignment honours rulings, same as the
-- monitor's GET /api/sessions. Dropped first: create-or-replace cannot change
-- an output column's expression type safely across versions.
drop view if exists public.sessions_v;
create view public.sessions_v
with (security_invoker = true) as
select i.id, i.text, i.started_at, i.ended_at, i.status, i.llm_category,
       i.project_id,
       p.name as project_name,
       (select avg(coalesce(case f.human_align when 1 then 1.0 when 0 then 0.0 end,
                            f.align_score))
          from public.focus_logs f where f.intention_id = i.id) as avg_align,
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
