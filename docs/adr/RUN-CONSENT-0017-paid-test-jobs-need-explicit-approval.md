# ADR-RUN-CONSENT-0017: Run on Cloud paid test jobs need explicit approval

Date: 2026-11-18

## Status

Proposed

## Context

The skill treated a Run on Cloud handoff as permission to spend credits, while
the frontend prohibited every job. Users need a saved workflow link and the
choice of paying for a test before opening it.

## Decision

The skill asks for explicit approval of one paid job, including validation or
preflight jobs. Upload, swap, build and deployment approvals keep their existing
scope. A declined or unanswered test request skips job-submitting commands,
including `deploy run --no-upload`, and still saves the prepared editor workflow
against the selected target. The report distinguishes unverified preparation
from successful execution. A timeout watches the original job; another paid job
requires another approval.

## Alternatives and consequences

Implicit approval avoids a question but can spend money unexpectedly. A blanket
execution ban prevents testing even when requested. Explicit job consent adds a
chat turn and supports both outcomes. It is an agent instruction, not a new
CLI-wide enforcement mechanism. The frontend handoff delegates execution and
reporting policy to the installed skill, so this behavior requires the updated
skill rather than a duplicated frontend rule.

## Validation

Skill contract tests and local agent scenarios cover stock Cloud, unverified
models, custom deployments, declined tests, partner charges and timeout recovery.
