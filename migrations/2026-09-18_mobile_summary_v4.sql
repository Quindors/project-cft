-- mobile_summary v4: derive session_active from `intentions`, not focus_logs.
--
-- Why: intentions.start() pushes the row (status='active') to Supabase on a
-- background thread the moment a session is declared — it lands in ~1s. The
-- old signal waited for a focus_logs row to carry intention_id, which costs
-- decision_interval (10s) + idle_flush_sec (15s) before it even reaches the
-- cloud. Reading intentions removes that ~25s of lag from the phone's lock.
--
-- It also fixes the trailing edge: the old "any intention row in the last 2
-- minutes" test left the lock up for up to 2 min after a session ended.
-- status='active' flips the instant the session closes.
--
-- Two guards keep the lock from sticking on:
--   * ends_at must still be in the future, so a session the monitor never got
--     to close (crash, power loss) expires on its own schedule.
--   * monitor_online must hold, preserving the old behaviour where the lock
--     releases if the PC stops reporting. This costs no latency: the monitor
--     writes focus_logs rows continuously whether or not a session is running.
--
-- started_at/ends_at are TEXT in zero-padded 'YYYY-MM-DD HH24:MI:SS' local
-- wall-clock form (see the wall-clock notes), so string comparison IS
-- chronological comparison — compared as text on purpose, since a cast would
-- raise and take the whole summary down if any row were ever malformed.
-- Replaces v3 (same signature, safe to re-run).

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
    select coalesce(nullif(trim(human_label), ''), category_name) as category_name
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
  ),
  online as (
    select coalesce(
      (select latest.wall > nowl.now_local - interval '90 seconds' from latest, nowl),
      false
    ) as is_online
  ),
  live_session as (
    select exists (
      select 1
      from intentions, nowl
      where intentions.status = 'active'
        and intentions.ends_at is not null
        and intentions.ends_at > to_char(nowl.now_local, 'YYYY-MM-DD HH24:MI:SS')
    ) as is_live
  )
  select jsonb_build_object(
    'today_samples', (select count(*)::int from today),
    'categories', (select coalesce(jsonb_agg(jsonb_build_object('name', category_name, 'samples', samples) order by samples desc), '[]'::jsonb) from cats),
    'last_seen', (select wall from latest),
    'last_category', (select category_name from latest),
    'monitor_online', (select is_online from online),
    'session_active', (select is_online from online) and (select is_live from live_session),
    'align_score', (select align_score from latest)
  );
$$;

grant execute on function public.mobile_summary(text) to anon, authenticated;
