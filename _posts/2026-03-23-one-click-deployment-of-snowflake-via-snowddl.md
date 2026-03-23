---
layout: post
title: One-click deployment of Snowflake via SnowDDL
description: Time to stop clicking/switching tabs around in Snowflake. Struggling
  about setting up RBAC, database and table layer? Deploy your desirable objects as
  YAML as easier
image:
category:
- Data Engineering
tags: snowflake snowddl yaml iac devops
date: 2026-03-23 23:41 +0100
---
## Intro

If you've spent time managing a Snowflake account at scale, you've felt the pain. A new engineer/analyst joins, we have to ask someone manually creates a user, assigns a role, and grants warehouse access. Three months later, nobody remembers what was granted or why (I admitted it was me). Multiply that by 100+ users and a dozen warehouses, and you've got configuration drift with no audit trail.

That's exactly the problem I ran into rebuilding Enpal Energy's data infrastructure. The fix? **SnowDDL** — a declarative, YAML-based object management tool that brings real infrastructure-as-code to Snowflake.

This post walks you through getting started with it. Let's have it a try!

---

## Why Not Terraform?

Terraform works, and indeed it was utilized in Enpal's master Snowflake data platform, but it's too heavy to maintain by smaller team, especially for Snowflake-specific workflows. The provider has coverage gaps, keeping state in sync is finicky, and making it truly DRY takes a lot of module wrangling. And, the learning curve is not friendly to colleagues with Data Scientist background. In contrary, SnowDDL is purpose-built for Snowflake, no state files, no provider quirks, just YAML configs that describe what your account should look like and it's already familiar to most of my team members. Looks like a perfect match!

---

## How SnowDDL Works

SnowDDL is **declarative**. You define the desired state; it compares against what's live and applies only the diff. If something exists in Snowflake but not in your config, it gets dropped. This makes your repo the single source of truth.

### 1. Install & Setup

Install `snowddl` according to your preferences via `pip`, `pipx`, `uv`

Create a dedicated service user using `ACCOUNTADMIN` (don't run SnowDDL as `ACCOUNTADMIN` in production)

```sql
USE ROLE ACCOUNTADMIN;
CREATE ROLE SNOWDDL_ADMIN;

GRANT ROLE SYSADMIN TO ROLE SNOWDDL_ADMIN;
GRANT ROLE SECURITYADMIN TO ROLE SNOWDDL_ADMIN;

CREATE USER SNOWDDL
TYPE = SERVICE
RSA_PUBLIC_KEY = 'rsa_public_key'
DEFAULT_ROLE = SNOWDDL_ADMIN;

GRANT ROLE SNOWDDL_ADMIN TO USER SNOWDDL;
GRANT ROLE SNOWDDL_ADMIN TO ROLE ACCOUNTADMIN;
```

### 2. Directory Structure

SnowDDL's config layout mirrors Snowflake's object hierarchy:

```
snowflake_config/
├── warehouse.yaml          # All warehouses
├── user.yaml               # All users (human and services)
├── business_role.yaml      # Business roles (analyst, engineer, etc.)
├── technical_role.yaml     # Object's access role if necessary
├── analytics_db/
│   ├── params.yaml         # Database-level settings
│   ├── raw/
│   │   ├── params.yaml     # Schema settings
│   │   └── table/
│   │       └── events.yaml # Table definitions
│   └── marts/
│       └── table/
│           └── revenue.yaml
```

Account-level objects (warehouses, users, roles) live as single YAML files in the root. Schema-level objects (tables, views, tasks) nest under `<database>/<schema>/<object_type>/`.

---

## Core Config Examples

### Warehouses

```yaml
# warehouse.yaml
analytics_wh:
  size: MEDIUM
  auto_suspend: 60
  comment: "General analytics workload"

etl_wh:
  size: LARGE
  auto_suspend: 120
  comment: "ELT pipeline warehouse"
```

All warehouses are created with `INITIALLY_SUSPENDED` and `AUTO_RESUME` enabled by default — no extra config needed.

### Business Roles

This is where SnowDDL shines. Instead of manually scripting `GRANT` statements, you describe access in plain terms:

```yaml
# business_role.yaml
analyst:
  schema_read:
    - analytics_db.marts
  warehouse_usage:
    - analytics_wh
  comment: "Read access to marts, analytics warehouse"

data_engineer:
  schema_write:
    - analytics_db.raw
    - analytics_db.marts
  warehouse_usage:
    - etl_wh
    - analytics_wh
  comment: "Write access to all schemas"
```

SnowDDL automatically generates a 3-tier role hierarchy under the hood: schema-level access roles → business roles → user roles. You just wire them together.

### Users

```yaml
# user.yaml
jane.smith:
  first_name: "Jane"
  last_name: "Smith"
  email: "jane@company.com"
  business_roles:
    - analyst

etl_service:
  rsa_public_key: "MIIBIjANBgkq..."
  business_roles:
    - data_engineer
  comment: "Service account for Fivetran"
```

---

## Running It

```bash
# Preview changes
snowddl plan -c ./snowflake_config --env-prefix DEV -a <account> \
  -u SNOWDDL -k path_to_private_key

# Apply safe DDL, e.g. CREATE
snowddl apply -c ./snowflake_config --env-prefix DEV -a <account> \
  -u SNOWDDL -k path_to_private_key
  
# Apply unsafe DDL, e.g. ALTER, DROP
snowddl apply -c ./snowflake_config --env-prefix DEV -a <account> \
  -u SNOWDDL -k path_to_private_key --apply-unsafe
```

The `--env-prefix` flag is key. It namespaces every object (e.g., `DEV__ANALYTICS_DB`, 2 underscores (__) as the default separator), so you can safely run the same config against dev, staging, and production accounts.

---

## CI/CD Integration

Wire it into GitHub Actions for hands-off deployments:

```yaml
# .github/workflows/snowflake-deploy.yml
on:
  push:
    branches: [main]
    paths: ['snowflake/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install snowddl
      - run: |
          snowddl apply -c ./snowflake_config \
            --env-prefix PROD \
            -a ${{ secrets.SF_ACCOUNT }} \
            -u ${{ secrets.SF_USER }} \
            -k ${{ secrets.SF_PRIVATE_KEY }}
```

Every infrastructure change becomes a PR. No more undocumented `GRANT` statements running ad hoc in the UI.

---

## What Gets You

A few things to know before you go all-in:

- **Dropped objects**: If you remove something from config, SnowDDL drops it. Use `snowddl plan` in `dev` to preview destructive changes.
- **Password changes**: SnowDDL can't compare existing password hashes, so credential rotation needs the `--refresh-user-passwords` flag.
- **Opinionated roles**: SnowDDL's 3-tier model works great for most setups. If you have highly custom RBAC needs, plan your mapping upfront.

---

## The Bottom Line

SnowDDL gave us version-controlled, reviewable, reproducible Snowflake infrastructure. Onboarding a new team member is a YAML entry and a PR. Role audits are a `git log`. Environment parity is a flag.

If you're managing Snowflake at any real scale and still doing it through the UI or ad hoc SQL, this is the tool to evaluate first.

**Resources:**
- [SnowDDL Docs](https://docs.snowddl.com)
- [GitHub — littleK0i/SnowDDL](https://github.com/littleK0i/SnowDDL)
