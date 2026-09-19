-- mobile_summary v3: match the dashboard's category math — a human correction
-- (human_label) overrides the AI category everywhere, exactly like
-- focus-dashboard's dataSource.js remaps rows before tallying. Without this,
-- corrected rows count under their original AI label and the phone's per-
-- category times drift from the dashboard's.
-- Replaces v2 (same signature, safe to re-run).

create or replace function public.mobile_summary(tz text default 'America/Los_Angeles')
returns jsonb
language sql
stable
set search_path = public
as $$
  with nowl as (
    select (now() at time zone tz) as now_local
  ),
  today as (
    select coalesce(nullif(trim(human_label), ''), category_name) as category_name, intention_id
    from focus_logs, nowl
    where (timestamp at time zone 'UTC') >= date_trunc('day', nowl.now_local)
  ),
  cats as (
    select category_name, count(*)::int as samples
    from today
    where category_name is not null
    group by category_name
  ),
  latest as (
    select (timestamp at time zone 'UTC') as wall,
           coalesce(nullif(trim(human_label), ''), category_name) as category_name,
           align_score
    from focus_logs
    order by timestamp desc
    limit 1
  )
  select jsonb_build_object(
    'today_samples', (select count(*)::int from today),
    'categories', (select coalesce(jsonb_agg(jsonb_build_object('name', category_name, 'samples', samples) order by samples desc), '[]'::jsonb) from cats),
    'last_seen', (select wall from latest),
    'last_category', (select category_name from latest),
    'monitor_online', coalesce((select latest.wall > nowl.now_local - interval '90 seconds' from latest, nowl), false),
    'session_active', exists(
      select 1 from focus_logs, nowl
      where (timestamp at time zone 'UTC') > nowl.now_local - interval '2 minutes'
        and intention_id is not null
    ),
    'align_score', (select align_score from latest)
  );
$$;

grant execute on function public.mobile_summary(text) to anon, authenticated;
