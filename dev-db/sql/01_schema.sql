CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    external_id UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    company_name TEXT,
    industry TEXT,
    segment TEXT,
    country TEXT,
    employee_band TEXT,
    signup_date DATE,
    is_active BOOLEAN,
    primary_contact_email TEXT,
    raw_source TEXT,
    notes TEXT
);

CREATE TABLE subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    plan_code TEXT,
    plan_name TEXT,
    mrr_usd NUMERIC(12, 2),
    seats INTEGER,
    billing_cycle TEXT,
    start_date DATE,
    cancelled_at DATE,
    status TEXT,
    contract_term_months INTEGER,
    cancellation_reason TEXT,
    imported_from TEXT
);

CREATE TABLE usage_monthly (
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    month_start DATE NOT NULL,
    active_users INTEGER,
    api_requests BIGINT,
    storage_gb NUMERIC(10, 2),
    feature_adoption_score NUMERIC(5, 4),
    PRIMARY KEY (customer_id, month_start)
);

CREATE TABLE invoices (
    invoice_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    subscription_id INTEGER REFERENCES subscriptions (subscription_id),
    invoice_date DATE,
    due_date DATE,
    paid_date DATE,
    amount_usd NUMERIC(12, 2),
    status TEXT,
    payment_method TEXT,
    invoice_number TEXT,
    memo TEXT
);

CREATE TABLE support_tickets (
    ticket_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    category TEXT,
    priority TEXT,
    csat_score SMALLINT,
    subject TEXT,
    body TEXT
);

CREATE TABLE nps_responses (
    response_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    survey_date DATE,
    score SMALLINT,
    freeform_comment TEXT
);

CREATE TABLE account_events (
    event_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    occurred_at TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    source_system TEXT,
    payload JSONB,
    raw_message TEXT
);

CREATE INDEX idx_subscriptions_customer_id ON subscriptions (customer_id);
CREATE INDEX idx_subscriptions_status ON subscriptions (status);
CREATE INDEX idx_usage_monthly_month_start ON usage_monthly (month_start);
CREATE INDEX idx_invoices_customer_id ON invoices (customer_id);
CREATE INDEX idx_invoices_status ON invoices (status);
CREATE INDEX idx_support_tickets_customer_id ON support_tickets (customer_id);
CREATE INDEX idx_nps_responses_customer_id ON nps_responses (customer_id);
CREATE INDEX idx_account_events_customer_id ON account_events (customer_id);
CREATE INDEX idx_account_events_occurred_at ON account_events (occurred_at);
CREATE INDEX idx_account_events_event_type ON account_events (event_type);

COMMENT ON TABLE customers IS 'B2B SaaS customer accounts';
COMMENT ON TABLE account_events IS 'Raw account lifecycle and operational events (messy by design)';
