# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Mutual Dissent, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email **security@q-uestionable.ai** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
3. Allow up to 72 hours for initial response
4. We will coordinate disclosure timeline with you

## Scope

Mutual Dissent is a multi-model debate engine. Vulnerabilities in the tool itself are in scope:

- API key leakage in transcripts, logs, or web UI
- Command injection in CLI argument handling
- Prompt injection via web UI inputs
- Unsafe deserialization of transcript data
- Dependency vulnerabilities with exploitable paths

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
