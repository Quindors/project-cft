-- Enable RLS on focus_logs + categories, scoped so every existing client keeps working.
--
-- Current clients all use the anon key (exe via .env, dashboard via VITE_SUPABASE_ANON_KEY,
-- phone app via EXPO_PUBLIC_SUPABASE_ANON_KEY), so policies target anon. What each needs:
--   exe:         insert focus_logs; delete [ERROR]/[PARSE_ERROR] rows; select categories
--   dashboard:   select both; update focus_logs.human_label/category_name; manage categories
--   supa_mirror: update focus_logs.human_label/category_name; upsert + delete categories
--   phone:       select both; call mobile_summary()
--
-- What this closes: arbitrary UPDATE (timestamps, windows, typed_text), arbitrary DELETE,
-- and TRUNCATE via the anon key. Reads stay public-with-key until Supabase Auth lands
-- (dashboard + phone would need sign-in first — see repo notes).
--
-- NOTE: if the reviewed_at column migration is ever applied to Supabase, extend the
-- column grant: grant update (human_label, category_name, reviewed_at) ...

alter table public.focus_logs enable row level security;
alter table public.categories enable row level security;

create policy focus_logs_select on public.focus_logs for select to anon, authenticated using (true);
create policy focus_logs_insert on public.focus_logs for insert to anon, authenticated with check (true);
create policy focus_logs_update on public.focus_logs for update to anon, authenticated using (true) with check (true);
create policy focus_logs_delete on public.focus_logs for delete to anon, authenticated using (category_name in ('[ERROR]','[PARSE_ERROR]'));

revoke update, truncate on table public.focus_logs from anon, authenticated;
grant update (human_label, category_name) on public.focus_logs to anon, authenticated;

create policy categories_select on public.categories for select to anon, authenticated using (true);
create policy categories_insert on public.categories for insert to anon, authenticated with check (true);
create policy categories_update on public.categories for update to anon, authenticated using (true) with check (true);
create policy categories_delete on public.categories for delete to anon, authenticated using (true);
revoke truncate on table public.categories from anon, authenticated;

-- One-round-trip summary for the phone app. security invoker: RLS applies.
-- session_active keys off intention_id, which the exe only writes after its
-- next rebuild (repo code already does; the frozen exe predates it).
create or replace function public.mobile_summary(tz text default 'America/New_York')
returns jsonb
language sql
stable
set search_path = public
as $$
  with bounds as (
    select (date_trunc('day', now() at time zone tz) at time zone tz) as day_start
  ),
  today as (
    select category_name, intention_id
    from focus_logs, bounds
    where timestamp >= bounds.day_start
  ),
  cats as (
    select category_name, count(*)::int as samples
    from today
    where category_name is not null
    group by category_name
  ),
  latest as (
    select timestamp, category_name, align_score
    from focus_logs
    order by timestamp desc
    limit 1
  )
  select jsonb_build_object(
    'today_samples', (select count(*)::int from today),
    'categories', (select coalesce(jsonb_agg(jsonb_build_object('name', category_name, 'samples', samples) order by samples desc), '[]'::jsonb) from cats),
    'last_seen', (select timestamp from latest),
    'last_category', (select category_name from latest),
    'monitor_online', coalesce((select timestamp > now() - interval '90 seconds' from latest), false),
    'session_active', exists(select 1 from focus_logs where timestamp > now() - interval '2 minutes' and intention_id is not null),
    'align_score', (select align_score from latest)
  );
$$;

grant execute on function public.mobile_summary(text) to anon, authenticated;
