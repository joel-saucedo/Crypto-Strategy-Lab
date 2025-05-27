#!/bin/bash

# Crypto-Strategy-Lab: Git automation script
# Handles both initial repo creation and subsequent pushes

# Variables - Replace these with your desired values
REPO_NAME="Crypto-Strategy-Lab"
COMMIT_MESSAGE="Add core configuration files and backtest engine"

# Check if this is already a git repository
if [ ! -d ".git" ]; then
    echo "Initializing new Git repository..."
    # First-time setup: initialize, create repo, and push
    git init && \
    git branch -M main && \
    git add . && \
    git commit -m "$COMMIT_MESSAGE" && \
    gh repo create "$REPO_NAME" --public --source=. --push
    
    if [ $? -eq 0 ]; then
        echo "Successfully initialized Git repository, created GitHub repository '$REPO_NAME', and pushed to main branch."
    else
        echo "Error during initial setup. Check your GitHub CLI authentication and try again."
        exit 1
    fi
else
    echo "Repository already exists. Adding changes and pushing..."
    # Subsequent updates: add, commit, and push
    git add . && \
    git commit -m "$COMMIT_MESSAGE" && \
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo "Successfully committed changes and pushed to GitHub."
    else
        echo "Error during push. Check your remote configuration and try again."
        exit 1
    fi
fi

# Optional: Show current status
echo "Current repository status:"
git status --short