# GitHub Setup

This folder is already initialized as a local git repository on branch `main`.

`gh` is not installed on the current cluster image, so create the remote from a machine/session with GitHub CLI or through the GitHub web UI.

## With GitHub CLI

```bash
cd /scratch/gautschi/bansa125/storymem-agentic
git config user.name "Your Name"
git config user.email "you@example.com"
git add .
git commit -m "Initial StoryMem Agentic scaffold"
gh repo create storymem-agentic --private --source . --remote origin --push
```

## Without GitHub CLI

1. Create a private empty repo named `storymem-agentic` on GitHub.
2. Copy the SSH or HTTPS remote URL.
3. Run:

```bash
cd /scratch/gautschi/bansa125/storymem-agentic
git config user.name "Your Name"
git config user.email "you@example.com"
git add .
git commit -m "Initial StoryMem Agentic scaffold"
git remote add origin <REMOTE_URL>
git push -u origin main
```

Keep models, checkpoints, results, logs, caches, generated media, and `.venv/` out of git.
