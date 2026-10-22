-- Read-only role for datasynth (run after schema, data, and views).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'datasynth_ro') THEN
        CREATE ROLE datasynth_ro LOGIN PASSWORD 'datasynth_ro_dev';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE churn_dev TO datasynth_ro;
GRANT USAGE ON SCHEMA public TO datasynth_ro;

GRANT SELECT ON ALL TABLES IN SCHEMA public TO datasynth_ro;
GRANT REFERENCES ON ALL TABLES IN SCHEMA public TO datasynth_ro;

ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO datasynth_ro;
