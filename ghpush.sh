#!/bin/bash

# Simple git push script - automatically commits and pushes
COMMIT_MESSAGE="[test] Fix strategy unit tests - restore comprehensive test coverage"

# Show current changes
echo "Current changes:"
git status --short

echo "Committing with message: $COMMIT_MESSAGE"

# Add, commit, and push
git add . && \
git commit -m "$COMMIT_MESSAGE" && \
git push

if [ $? -eq 0 ]; then
    echo "Successfully pushed changes to GitHub."
else
    echo "Error during push."
    exit 1
fi