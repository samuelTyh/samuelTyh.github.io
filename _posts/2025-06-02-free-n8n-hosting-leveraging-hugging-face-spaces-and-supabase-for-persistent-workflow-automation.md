---
layout: post
title: 'Free n8n Hosting: Leveraging Hugging Face Spaces and Supabase for Persistent Workflow Automation'
description: n8n Cloud deployment method that beginners can use for free
image: 
category: [Learning Journey]
tags: n8n automation workflow huggingface supabase
date: 2025-06-02 20:52 +0200
---

## Introduction

n8n, the fair-code workflow automation platform, has transformed how developers and businesses automate their processes. While self-hosting n8n typically requires dedicated infrastructure, this guide demonstrates an innovative approach: hosting n8n entirely free using Hugging Face Spaces for compute and Supabase for persistent storage.

This architecture addresses a critical limitation of Hugging Face's free tier—ephemeral storage—by offloading workflow persistence to Supabase's generous free database tier. The result is a production-ready n8n instance without infrastructure costs, perfect for individuals, small teams, or proof-of-concept deployments.

## Architecture Overview

The solution leverages two complementary platforms:

**Hugging Face Spaces** provides:
- Free Docker container hosting
- 2 vCPU and 16GB RAM (free tier)
- Public HTTPS endpoint
- Automatic container management

**Supabase** delivers:
- PostgreSQL database (500MB free tier)
- Row-level security
- Real-time subscriptions
- Built-in authentication

This combination circumvents Hugging Face's ephemeral storage limitation while maintaining the performance characteristics needed for reliable workflow automation.

## Prerequisites

Before beginning, ensure you have:
- A Hugging Face account (free signup at huggingface.co)
- A Supabase account (free signup at supabase.com)
- Basic familiarity with Docker and environment variables
- Git installed locally

## Step 1: Set Up Supabase Database

First, create the PostgreSQL backend that will store your n8n workflows, credentials, and execution history.

### Create a New Supabase Project

1. Log into your Supabase dashboard
2. Click "New project"
3. Configure your project:
   - **Name**: `n8n-backend`
   - **Database Password**: Generate a strong password (save this!)
   - **Region**: Choose the closest to your primary users
4. Wait for project provisioning (typically 2-3 minutes)

### Configure Database Schema

n8n requires specific database tables and permissions. Navigate to the SQL editor in your Supabase dashboard and execute:

```sql
-- Create n8n schema
CREATE SCHEMA IF NOT EXISTS n8n;

-- Grant permissions to authenticated users
GRANT ALL ON SCHEMA n8n TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA n8n TO postgres;
GRANT ALL ON ALL SEQUENCES IN SCHEMA n8n TO postgres;

-- Enable UUID extension for n8n
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### Retrieve Connection Details

From your Supabase project settings, collect:
- **Host**: Found under Settings → Database
- **Port**: Typically 5432
- **Database name**: Usually 'postgres'
- **User**: 'postgres'
- **Password**: The password you created earlier

Your connection string will look like:
```
postgresql://postgres:[YOUR-PASSWORD]@[YOUR-PROJECT-REF].supabase.co:5432/postgres
```

## Step 2: Create Hugging Face Space

Now, set up the compute environment for n8n.

### Initialize Space Repository

1. Navigate to huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Space name**: `n8n-workflow-automation`
   - **SDK**: Docker
   - **Visibility**: Public (required for free tier)
4. Clone the repository locally:
```bash
git clone https://huggingface.co/spaces/[YOUR-USERNAME]/n8n-workflow-automation
cd n8n-workflow-automation
```

### Create Dockerfile

Create a `Dockerfile` that configures n8n with Supabase integration:

```dockerfile
FROM n8nio/n8n:latest

# Install PostgreSQL client for better database compatibility
USER root
RUN apk add --no-cache postgresql-client

# Create data directory with proper permissions
RUN mkdir -p /home/node/.n8n && \
    chown -R node:node /home/node/.n8n

USER node

# Set working directory
WORKDIR /home/node

# Expose n8n port
EXPOSE 5678

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:5678/healthz || exit 1

# Start n8n
CMD ["n8n", "start"]
```

### Configure Environment Variables

Create a `.env` file template (don't commit actual values):

```bash
# Database Configuration
DB_TYPE=postgresdb
DB_POSTGRESDB_HOST=your-project.supabase.co
DB_POSTGRESDB_PORT=5432
DB_POSTGRESDB_DATABASE=postgres
DB_POSTGRESDB_USER=postgres
DB_POSTGRESDB_PASSWORD=your-password
DB_POSTGRESDB_SCHEMA=n8n

# n8n Configuration
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=your-admin-password
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_PROTOCOL=https
WEBHOOK_URL=https://your-space.hf.space

# Execution Configuration
EXECUTIONS_MODE=regular
EXECUTIONS_PROCESS=main
N8N_METRICS=false

# Security
N8N_ENCRYPTION_KEY=your-32-char-encryption-key
```

### Create Space Configuration

Add a `README.md` with YAML frontmatter:

```markdown
---
title: n8n Workflow Automation
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# n8n Workflow Automation Platform

This Space hosts a fully functional n8n instance with persistent storage via Supabase.

## Features
- Complete n8n workflow automation
- Persistent workflow storage
- Secure credential management
- Webhook support
- API integrations

## Access
Visit the Space URL and log in with the configured credentials.
```

## Step 3: Deploy and Configure

### Set Hugging Face Secrets

In your Space settings, add these secrets:
1. Navigate to Settings → Variables and secrets
2. Add each environment variable from your `.env` file
3. Mark sensitive values (passwords, keys) as "Secret"

### Initial Deployment

Push your configuration:

```bash
git add .
git commit -m "Initial n8n configuration"
git push origin main
```

Hugging Face will automatically build and deploy your container. Monitor the logs for any issues.

### Verify Database Connection

Once deployed, access your n8n instance and verify:
1. Navigate to `https://[your-space].hf.space`
2. Log in with your basic auth credentials
3. Create a test workflow
4. Restart the Space (Settings → Restart)
5. Confirm your workflow persists

## Step 4: Configure n8n for Production Use

### Enable Webhook URLs

For webhook-triggered workflows, configure the public URL:

1. In n8n settings, set Webhook URL to your Space URL
2. Test with a simple webhook workflow:
```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300],
      "webhookId": "test-webhook",
      "parameters": {
        "path": "test",
        "responseMode": "onReceived",
        "responseData": "allEntries"
      }
    }
  ]
}
```

### Set Up Credentials Encryption

n8n encrypts stored credentials using the `N8N_ENCRYPTION_KEY`. Generate a secure key:

```bash
openssl rand -hex 32
```

Update this in your Space secrets to ensure credential security.

### Configure Execution Retention

To manage database growth within Supabase's free tier, configure execution pruning:

```bash
EXECUTIONS_DATA_SAVE_ON_ERROR=all
EXECUTIONS_DATA_SAVE_ON_SUCCESS=none
EXECUTIONS_DATA_SAVE_MANUAL_EXECUTIONS=true
EXECUTIONS_DATA_MAX_AGE=168  # 7 days
```

## Optimization and Best Practices

### Performance Tuning

1. **Connection Pooling**: Configure n8n's database connection pool:
```bash
DB_POSTGRESDB_POOL_MIN=2
DB_POSTGRESDB_POOL_MAX=10
```

2. **Memory Management**: Monitor Space metrics and adjust workflow complexity accordingly

3. **Webhook Response Times**: Keep webhook workflows lightweight to avoid timeouts

### Security Considerations

1. **Access Control**: Always enable basic authentication or implement OAuth
2. **Network Security**: Use Supabase's connection pooler for additional security
3. **Credential Rotation**: Regularly update passwords and API keys
4. **Audit Logging**: Enable n8n's audit logs for compliance

### Backup Strategy

Implement regular backups despite the free tier limitations:

```bash
# Weekly backup script (run locally)
pg_dump -h your-project.supabase.co -U postgres -d postgres -n n8n > backup_$(date +%Y%m%d).sql
```

## Advantages for n8n Beginners

This setup offers several compelling benefits:

### Zero Infrastructure Costs
- No server hosting fees
- No database hosting costs
- No domain or SSL certificate expenses
- Perfect for learning and experimentation

### Production-Ready Features
- HTTPS endpoint provided automatically
- Database backups via Supabase
- Scalable to paid tiers when needed
- Professional deployment practices

### Learning Opportunities
- Understand containerized deployments
- Practice with PostgreSQL databases
- Explore webhook integrations
- Build real automation workflows

### Easy Migration Path
When ready to scale:
1. Export workflows from n8n
2. Backup Supabase database
3. Deploy to any cloud provider
4. Import data and continue

## Troubleshooting Common Issues

### Space Sleeping
Hugging Face Spaces sleep after inactivity. Solutions:
- Use external monitoring to ping your Space
- Implement a scheduled workflow that runs regularly
- Upgrade to a paid Space for always-on availability

### Database Connection Errors
If n8n can't connect to Supabase:
1. Verify connection string formatting
2. Check Supabase connection limits
3. Ensure proper SSL mode configuration
4. Review Space logs for detailed errors

### Webhook Timeouts
For long-running webhooks:
- Implement async processing patterns
- Use n8n's "Respond to Webhook" node
- Break complex workflows into smaller pieces

## Future Enhancements

As you grow comfortable with this setup, consider:

1. **Custom Nodes**: Build and deploy custom n8n nodes
2. **Multi-Instance**: Run multiple n8n instances with shared database
3. **Advanced Monitoring**: Integrate with Supabase's real-time features
4. **API Gateway**: Add rate limiting and authentication layers

## Conclusion

Hosting n8n on Hugging Face Spaces with Supabase backend represents a paradigm shift in accessible workflow automation. This architecture eliminates traditional barriers to entry while maintaining professional-grade capabilities. 

For beginners, it provides a risk-free environment to explore automation possibilities. For experienced users, it offers a viable production platform for non-critical workflows. As the ecosystem evolves, expect tighter integrations and enhanced capabilities that further democratize workflow automation.

The convergence of specialized platforms like Hugging Face and Supabase demonstrates the future of composable infrastructure—where developers assemble best-in-class services rather than managing monolithic deployments. This approach not only reduces operational overhead but accelerates innovation by allowing focus on business logic rather than infrastructure management.

Start building your automation workflows today, and join the growing community leveraging free-tier infrastructure for real-world solutions.
