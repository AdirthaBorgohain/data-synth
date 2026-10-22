"""Wipe, recreate schema, and load synthetic B2B SaaS churn data."""

from __future__ import annotations

import os
import random
import sys
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import psycopg
from faker import Faker
from psycopg.types.json import Json
from psycopg import sql

N_CUSTOMERS = 4_000
N_MONTHS = 24
REFERENCE_DATE = date(2026, 5, 1)
RANDOM_SEED = 42

INDUSTRIES = (
    "Software",
    "Financial Services",
    "Healthcare",
    "Retail",
    "Manufacturing",
    "Education",
    "Professional Services",
    "Media",
)
SEGMENTS = ("smb", "mid_market", "enterprise")
EMPLOYEE_BANDS = ("1-10", "11-50", "51-200", "201-1000", "1000+")
PLAN_CODES = ("starter", "professional", "business", "enterprise")
PLAN_MRR = {
    "starter": (49, 199),
    "professional": (199, 799),
    "business": (799, 2499),
    "enterprise": (2499, 12000),
}
CHURN_REASONS = (
    "voluntary_cancel",
    "non_payment",
    "downgrade_to_free",
    "acquisition_rollup",
)
TICKET_CATEGORIES = (
    "billing",
    "technical",
    "onboarding",
    "feature_request",
    "account",
)
TICKET_PRIORITIES = ("low", "medium", "high", "critical")


def _admin_url() -> str:
    url = os.environ.get("ADMIN_DATABASE_URL")
    if not url:
        raise RuntimeError("ADMIN_DATABASE_URL is required.")
    return url


def _sql_dir() -> Path:
    return Path(os.environ.get("SQL_DIR", "/sql"))


def _run_sql_file(conn: psycopg.Connection, path: Path) -> None:
    body = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(body)
    conn.commit()
    print(f"  applied {path.name}")


def _wipe_and_apply_schema(conn: psycopg.Connection) -> None:
    sql_root = _sql_dir()
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE;")
        cur.execute("CREATE SCHEMA public;")
        cur.execute("GRANT ALL ON SCHEMA public TO churn_admin;")
        cur.execute("GRANT ALL ON SCHEMA public TO public;")
    conn.commit()

    _run_sql_file(conn=conn, path=sql_root / "01_schema.sql")


def _month_starts() -> list[date]:
    first = REFERENCE_DATE.replace(day=1) - timedelta(days=32 * (N_MONTHS - 1))
    first = first.replace(day=1)
    months: list[date] = []
    current = first
    for _ in range(N_MONTHS):
        months.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _plan_for_segment(rng: random.Random, segment: str) -> str:
    weights = {
        "smb": [0.45, 0.35, 0.15, 0.05],
        "mid_market": [0.1, 0.35, 0.4, 0.15],
        "enterprise": [0.02, 0.08, 0.35, 0.55],
    }
    return rng.choices(PLAN_CODES, weights=weights[segment], k=1)[0]


def _generate_customers(rng: random.Random, fake: Faker) -> list[dict]:
    customers: list[dict] = []
    for _ in range(N_CUSTOMERS):
        segment = rng.choices(SEGMENTS, weights=[0.5, 0.35, 0.15], k=1)[0]
        tenure_months = int(rng.randint(3, 60))
        signup = REFERENCE_DATE - timedelta(days=tenure_months * 30 + rng.randint(0, 28))
        plan = _plan_for_segment(rng=rng, segment=segment)
        low, high = PLAN_MRR[plan]
        mrr = round(rng.uniform(low, high), 2)
        seats = {
            "starter": rng.randint(1, 10),
            "professional": rng.randint(5, 50),
            "business": rng.randint(20, 200),
            "enterprise": rng.randint(100, 2000),
        }[plan]

        support_load = rng.random()
        payment_risk = rng.random()
        usage_trend = rng.uniform(-0.3, 0.2)

        logit = (
            -2.68
            + 0.025 * max(0, 12 - tenure_months)
            + 0.75 * payment_risk
            + 0.45 * support_load
            - 0.6 * usage_trend
            + {"smb": 0.2, "mid_market": 0.05, "enterprise": -0.35}[segment]
            + {"starter": 0.15, "professional": 0.0, "business": -0.15, "enterprise": -0.3}[
                plan
            ]
        )
        churn_prob = _sigmoid(logit)
        will_churn = rng.random() < churn_prob

        if will_churn:
            churn_offset_days = rng.randint(14, min(tenure_months * 15, 180))
            churn_date = REFERENCE_DATE - timedelta(days=churn_offset_days)
            if churn_date <= signup:
                churn_date = signup + timedelta(days=rng.randint(60, 120))
        else:
            churn_date = None

        customers.append(
            {
                "external_id": str(uuid.UUID(int=rng.getrandbits(128))),
                "company_name": fake.company() if rng.random() > 0.01 else None,
                "industry": rng.choice(INDUSTRIES)
                if rng.random() > 0.08
                else rng.choice(["FinServ", "Health care", "", None]),
                "segment": segment
                if rng.random() > 0.06
                else rng.choice(["SMB", "mid market", "Enterprise", "", None]),
                "country": fake.country_code()[:2].upper()
                if rng.random() > 0.03
                else rng.choice(["USA", "UK", "DEU", "", None]),
                "employee_band": rng.choice(EMPLOYEE_BANDS)
                if rng.random() > 0.07
                else rng.choice(["1 to 10", "11- 50", "200+", "", None]),
                "signup_date": signup if rng.random() > 0.02 else None,
                "is_active": (churn_date is None) if rng.random() > 0.02 else None,
                "primary_contact_email": fake.company_email()
                if rng.random() > 0.03
                else rng.choice([fake.email(), "", None]),
                "raw_source": rng.choice(["salesforce", "zendesk", "stripe", "netsuite", None]),
                "notes": fake.sentence() if rng.random() < 0.2 else None,
                "plan_code": plan,
                "plan_name": plan.replace("_", " ").title(),
                "mrr_usd": mrr,
                "seats": seats,
                "billing_cycle": rng.choices(["monthly", "annual"], weights=[0.7, 0.3], k=1)[
                    0
                ],
                "will_churn": will_churn,
                "churn_date": churn_date,
                "churn_reason": rng.choice(CHURN_REASONS) if will_churn else None,
                "support_load": support_load,
                "payment_risk": payment_risk,
                "usage_trend": usage_trend,
            }
        )
    return customers


def _usage_rows(
    customer_id: int,
    months: list[date],
    base_users: int,
    base_api: int,
    churn_date: date | None,
    usage_trend: float,
    rng: random.Random,
) -> list[tuple]:
    rows: list[tuple] = []
    for idx, month_start in enumerate(months):
        if month_start < date(2020, 1, 1):
            continue
        months_to_end = len(months) - idx - 1
        decline = 1.0
        if churn_date is not None:
            months_before = (churn_date.year - month_start.year) * 12 + (
                churn_date.month - month_start.month
            )
            if 0 <= months_before <= 6:
                decline = 0.55 + 0.08 * months_before
            elif months_before < 0:
                decline = 0.45

        trend_factor = 1.0 + usage_trend * (idx / max(len(months) - 1, 1))
        noise = rng.uniform(0.85, 1.15)
        active_users = max(1, int(base_users * decline * trend_factor * noise))
        api_requests = max(
            100,
            int(base_api * decline * trend_factor * noise * rng.uniform(0.9, 1.1)),
        )
        storage = round(rng.uniform(5, 500) * decline * noise, 2)
        adoption = round(
            min(1.0, max(0.05, rng.uniform(0.3, 0.95) * decline * trend_factor)),
            4,
        )
        rows.append(
            (
                customer_id,
                month_start,
                active_users,
                api_requests,
                storage,
                adoption,
            )
        )
    return rows


def _insert_customers(conn: psycopg.Connection, customers: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO customers (
                external_id, company_name, industry, segment, country,
                employee_band, signup_date, is_active, primary_contact_email,
                raw_source, notes
            ) VALUES (
                %(external_id)s::uuid, %(company_name)s, %(industry)s, %(segment)s,
                %(country)s, %(employee_band)s, %(signup_date)s, %(is_active)s,
                %(primary_contact_email)s, %(raw_source)s, %(notes)s
            )
            """,
            [
                {
                    "external_id": c["external_id"],
                    "company_name": c["company_name"],
                    "industry": c["industry"],
                    "segment": c["segment"],
                    "country": c["country"],
                    "employee_band": c["employee_band"],
                    "signup_date": c["signup_date"],
                    "is_active": c["is_active"],
                    "primary_contact_email": c["primary_contact_email"],
                    "raw_source": c["raw_source"],
                    "notes": c["notes"],
                }
                for c in customers
            ],
        )
    conn.commit()


def _load_related(
    conn: psycopg.Connection, customers: list[dict], rng: random.Random, fake: Faker
) -> None:
    months = _month_starts()
    with conn.cursor() as cur:
        cur.execute("SELECT customer_id, signup_date FROM customers ORDER BY customer_id")
        id_rows = cur.fetchall()

    subscriptions: list[tuple] = []
    usage_rows: list[tuple] = []
    invoices: list[tuple] = []
    tickets: list[tuple] = []
    nps_rows: list[tuple] = []
    event_rows: list[tuple] = []

    for (customer_id, signup_date), meta in zip(id_rows, customers, strict=True):
        start_date = meta["signup_date"] or signup_date or (REFERENCE_DATE - timedelta(days=365))
        cancelled_at = meta["churn_date"]
        status = "cancelled" if cancelled_at else "active"
        if cancelled_at and (REFERENCE_DATE - cancelled_at).days < 30:
            status = "paused"

        subscriptions.append(
            (
                customer_id,
                meta["plan_code"] if rng.random() > 0.05 else rng.choice(["Pro", "ENT", "", None]),
                meta["plan_name"] if rng.random() > 0.08 else None,
                Decimal(str(meta["mrr_usd"])) if rng.random() > 0.03 else None,
                meta["seats"] if rng.random() > 0.04 else None,
                meta["billing_cycle"] if rng.random() > 0.04 else rng.choice(["mo", "yr", "", None]),
                start_date,
                cancelled_at,
                status if rng.random() > 0.03 else rng.choice(["Active", "CANCELLED", "", None]),
                12 if meta["billing_cycle"] == "monthly" else 12,
                meta["churn_reason"] if cancelled_at else (fake.word() if rng.random() < 0.05 else None),
                rng.choice(["stripe", "netsuite", "manual_csv", None]),
            )
        )

        if rng.random() < 0.12:
            subscriptions.append(
                (
                    customer_id,
                    rng.choice(["starter", "professional", "business", "enterprise", "Pro", None]),
                    None,
                    Decimal(str(round(float(meta["mrr_usd"]) * rng.uniform(0.7, 1.1), 2)))
                    if rng.random() > 0.2
                    else None,
                    meta["seats"] if rng.random() > 0.3 else None,
                    rng.choice(["monthly", "annual", "mo", None]),
                    start_date + timedelta(days=rng.randint(30, 180)),
                    cancelled_at if rng.random() < 0.5 else None,
                    rng.choice(["active", "cancelled", "paused", "trial", "Active", None]),
                    12,
                    None,
                    "manual_csv",
                )
            )

        base_users = max(2, (meta["seats"] or rng.randint(5, 100)) // rng.randint(2, 8))
        base_api = base_users * rng.randint(800, 4000)
        usage_rows.extend(
            _usage_rows(
                customer_id=customer_id,
                months=[m for m in months if m >= start_date.replace(day=1)],
                base_users=base_users,
                base_api=base_api,
                churn_date=meta["churn_date"],
                usage_trend=meta["usage_trend"],
                rng=rng,
            )
        )

        n_invoices = rng.randint(6, 28)
        for inv_i in range(n_invoices):
            inv_date = start_date + timedelta(days=inv_i * 30 + rng.randint(0, 10))
            if inv_date > REFERENCE_DATE:
                break
            amount = Decimal(str(round(float(meta["mrr_usd"]) * rng.uniform(0.9, 1.1), 2)))
            due = inv_date + timedelta(days=14)
            if meta["will_churn"] and meta["payment_risk"] > 0.55 and rng.random() < 0.35:
                inv_status = "failed"
                paid = None
            elif rng.random() < 0.08:
                inv_status = "open"
                paid = None
            else:
                inv_status = "paid"
                paid = due + timedelta(days=rng.randint(0, 7))
            invoice_number = f"INV-{customer_id:05d}-{inv_i:04d}" if rng.random() > 0.1 else None
            invoices.append(
                (
                    customer_id,
                    None,
                    inv_date if rng.random() > 0.02 else None,
                    due if rng.random() > 0.03 else None,
                    paid if rng.random() > 0.04 else None,
                    amount,
                    inv_status if rng.random() > 0.03 else rng.choice(["PAID", "Overdue", "", None]),
                    rng.choice(["card", "ach", "wire", "invoice", "cc", "", None]),
                    invoice_number,
                    fake.sentence() if rng.random() < 0.15 else None,
                )
            )

        n_tickets = int(rng.randint(0, 4) + meta["support_load"] * 12)
        if meta["will_churn"]:
            n_tickets += rng.randint(2, 8)
        for _ in range(n_tickets):
            span_days = max((REFERENCE_DATE - start_date).days, 1)
            opened = datetime.combine(
                start_date + timedelta(days=rng.randint(0, span_days)),
                datetime.min.time(),
            )
            closed = opened + timedelta(days=rng.randint(1, 14)) if rng.random() < 0.85 else None
            csat = None
            if closed and rng.random() < 0.7:
                base = 4 if not meta["will_churn"] else 2
                csat = max(1, min(5, base + rng.randint(-1, 1)))
            subject = fake.sentence(nb_words=6) if rng.random() > 0.08 else None
            body = fake.paragraph(nb_sentences=3) if rng.random() > 0.12 else None
            tickets.append(
                (
                    customer_id,
                    opened,
                    closed,
                    rng.choice(TICKET_CATEGORIES) if rng.random() > 0.06 else rng.choice(["tech", "bill", "", None]),
                    rng.choices(
                        TICKET_PRIORITIES,
                        weights=[0.4, 0.35, 0.2, 0.05],
                        k=1,
                    )[0]
                    if rng.random() > 0.05
                    else rng.choice(["P1", "sev2", "", None]),
                    csat if rng.random() > 0.07 else None,
                    subject,
                    body,
                )
            )

        for _ in range(rng.randint(1, 4)):
            survey = start_date + timedelta(
                days=rng.randint(90, max((REFERENCE_DATE - start_date).days, 91))
            )
            if survey > REFERENCE_DATE:
                continue
            base_nps = 8 if not meta["will_churn"] else 5
            nps_rows.append(
                (
                    customer_id,
                    survey,
                    max(0, min(10, base_nps + rng.randint(-2, 2))),
                    fake.sentence() if rng.random() < 0.55 else None,
                )
            )

        if meta["churn_date"]:
            event_rows.append(
                (
                    customer_id,
                    datetime.combine(meta["churn_date"], datetime.min.time()),
                    rng.choice(
                        [
                            "subscription.cancelled",
                            "account.closed",
                            "billing.write_off",
                            "status_change",
                        ]
                    ),
                    rng.choice(["crm", "billing", "support", "product"]),
                    Json({"reason": meta["churn_reason"], "free_text": fake.sentence()}),
                    None,
                )
            )
        if rng.random() < 0.15:
            effective_signup = meta["signup_date"] or start_date
            event_rows.append(
                (
                    customer_id,
                    datetime.combine(
                        effective_signup + timedelta(days=rng.randint(1, 120)),
                        datetime.min.time(),
                    ),
                    rng.choice(["onboarding.completed", "seat_change", "plan_change"]),
                    rng.choice(["crm", "product"]),
                    None,
                    fake.sentence(),
                )
            )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO subscriptions (
                customer_id, plan_code, plan_name, mrr_usd, seats, billing_cycle,
                start_date, cancelled_at, status, contract_term_months,
                cancellation_reason, imported_from
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            subscriptions,
        )
        cur.executemany(
            """
            INSERT INTO usage_monthly (
                customer_id, month_start, active_users, api_requests,
                storage_gb, feature_adoption_score
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            usage_rows,
        )
        cur.executemany(
            """
            INSERT INTO invoices (
                customer_id, subscription_id, invoice_date, due_date, paid_date,
                amount_usd, status, payment_method, invoice_number, memo
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            invoices,
        )
        cur.executemany(
            """
            INSERT INTO support_tickets (
                customer_id, opened_at, closed_at, category, priority, csat_score, subject, body
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            tickets,
        )
        cur.executemany(
            """
            INSERT INTO nps_responses (customer_id, survey_date, score, freeform_comment)
            VALUES (%s, %s, %s, %s)
            """,
            nps_rows,
        )
        cur.executemany(
            """
            INSERT INTO account_events (
                customer_id, occurred_at, event_type, source_system, payload, raw_message
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            event_rows,
        )
        cur.execute(
            """
            UPDATE invoices AS i
            SET subscription_id = s.subscription_id
            FROM subscriptions AS s
            WHERE i.customer_id = s.customer_id
            """
        )
    conn.commit()


def _print_summary(conn: psycopg.Connection) -> None:
    tables = (
        "customers",
        "subscriptions",
        "usage_monthly",
        "invoices",
        "support_tickets",
        "nps_responses",
        "account_events",
    )
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
            count = cur.fetchone()[0]
            print(f"  {table}: {count}")


def main() -> int:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    fake = Faker()
    Faker.seed(RANDOM_SEED)
    fake.seed_instance(RANDOM_SEED)

    print("Connecting ...")
    with psycopg.connect(_admin_url(), autocommit=False) as conn:
        print("Wiping and applying schema ...")
        _wipe_and_apply_schema(conn=conn)

        print("Generating customers ...")
        customers = _generate_customers(rng=rng, fake=fake)
        _insert_customers(conn=conn, customers=customers)

        print("Loading subscriptions, usage, invoices, tickets, NPS, events ...")
        _load_related(conn=conn, customers=customers, rng=rng, fake=fake)

        print("Applying grants ...")
        _run_sql_file(conn=conn, path=_sql_dir() / "00_roles.sql")

        print("Summary:")
        _print_summary(conn=conn)

    print("Seed complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
