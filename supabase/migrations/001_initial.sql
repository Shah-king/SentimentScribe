-- ─────────────────────────────────────────────────────────────────────────────
-- SentimentScribe — Initial Schema
-- Run this in the Supabase SQL Editor (supabase.com → project → SQL Editor)
-- ─────────────────────────────────────────────────────────────────────────────

-- ── predictions: one row per user inference ───────────────────────────────────
create table if not exists predictions (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users not null,
  review_text text not null,
  sentiment   text not null check (sentiment in ('positive', 'negative')),
  confidence  float not null check (confidence >= 0 and confidence <= 1),
  model_type  text not null default 'sklearn',
  created_at  timestamptz default now()
);
alter table predictions enable row level security;
create policy "users see own predictions"
  on predictions for all
  using (auth.uid() = user_id);

-- ── api_keys: hashed developer keys ──────────────────────────────────────────
create table if not exists api_keys (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid references auth.users not null,
  key_hash     text not null unique,
  name         text not null default 'My Key',
  created_at   timestamptz default now(),
  last_used_at timestamptz
);
alter table api_keys enable row level security;
create policy "users see own keys"
  on api_keys for all
  using (auth.uid() = user_id);

-- ── reports: shareable bulk analysis snapshots ───────────────────────────────
create table if not exists reports (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references auth.users,
  summary    jsonb not null,
  created_at timestamptz default now()
);
alter table reports enable row level security;
-- Anyone can read a report (needed for public share links)
create policy "public read reports"
  on reports for select
  using (true);
-- Only the owner can insert
create policy "owner can insert report"
  on reports for insert
  with check (auth.uid() = user_id);

-- ── bulk_jobs: CSV upload job tracking ────────────────────────────────────────
create table if not exists bulk_jobs (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid references auth.users not null,
  filename   text,
  row_count  int,
  report_id  uuid references reports,
  created_at timestamptz default now()
);
alter table bulk_jobs enable row level security;
create policy "users see own bulk jobs"
  on bulk_jobs for all
  using (auth.uid() = user_id);

-- ── Indexes for common queries ────────────────────────────────────────────────
create index if not exists idx_predictions_user_created
  on predictions (user_id, created_at desc);

create index if not exists idx_bulk_jobs_user_created
  on bulk_jobs (user_id, created_at desc);
