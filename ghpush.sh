#!/bin/bash

# Git push script for Crypto Strategy Lab
COMMIT_MESSAGE="Complete CLI suite with live monitoring dashboard, unified backtesting engine, and 12 validated strategies ready for production deployment"

# Show current changes
echo "Current changes:"
git status --short

echo ""
echo "Committing with message: $COMMIT_MESSAGE"
echo ""

# Add, commit, and push
git add . && \
git commit -m "$COMMIT_MESSAGE" && \
git push

if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully pushed changes to GitHub!"
else
    echo "Error during push."
    exit 1
fi