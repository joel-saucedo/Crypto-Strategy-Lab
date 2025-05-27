#!/bin/bash

# Crypto-Strategy-Lab: Git automation script
# Handles both initial repo creation and subsequent pushes

# Variables
REPO_NAME="Crypto-Strategy-Lab"
DEFAULT_COMMIT_MESSAGE="Update: Crypto Strategy Lab changes"

# Check if this is already a git repository
if [ ! -d ".git" ]; then
    echo "Initializing new Git repository..."
    # First-time setup: initialize, create repo, and push
    git init && \
    git branch -M main

    # Create and checkout develop branch
    git checkout -b develop
    
    # Prompt for commit message or use default
    read -p "Enter commit message (or press enter for default): " COMMIT_MESSAGE
    COMMIT_MESSAGE=${COMMIT_MESSAGE:-"Initial commit: Crypto-Strategy-Lab framework"}
    
    git add . && \
    git commit -m "$COMMIT_MESSAGE" && \
    gh repo create "$REPO_NAME" --public --source=. --push
    
    if [ $? -eq 0 ]; then
        echo "Successfully initialized Git repository, created GitHub repository '$REPO_NAME', and pushed to develop branch."
        echo "Creating main branch and pushing..."
        git checkout -b main
        git push -u origin main
        git checkout develop
        echo "Returned to develop branch for future work."
    else
        echo "Error during initial setup. Check your GitHub CLI authentication and try again."
        exit 1
    fi
else
    # Check current status
    echo "Current changes:"
    git status --short
    
    # Run tests before commit
    echo "Running tests to ensure code quality..."
    if command -v pytest &> /dev/null; then
        pytest src/strategies/*/test_signal.py -v
        TEST_STATUS=$?
        if [ $TEST_STATUS -ne 0 ]; then
            read -p "Tests failed. Do you want to continue with the commit? (y/N): " CONTINUE
            if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
                echo "Commit aborted. Please fix the tests before committing."
                exit 1
            fi
        else
            echo "All tests passed!"
        fi
    else
        echo "pytest not found, skipping tests."
    fi
    
    # Prompt for commit message with tag
    echo "Available tags: [feat], [fix], [docs], [style], [refactor], [test], [chore]"
    read -p "Enter commit message (or press enter for default): " COMMIT_MESSAGE
    COMMIT_MESSAGE=${COMMIT_MESSAGE:-"$DEFAULT_COMMIT_MESSAGE"}
    
    # Prompt for branch selection
    CURRENT_BRANCH=$(git branch --show-current)
    echo "Current branch: $CURRENT_BRANCH"
    
    if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "develop" ]; then
        # Feature branch flow
        git add . && \
        git commit -m "$COMMIT_MESSAGE" && \
        git push -u origin $CURRENT_BRANCH
        
        echo "Changes pushed to feature branch: $CURRENT_BRANCH"
        read -p "Do you want to create a pull request to develop? (y/N): " CREATE_PR
        if [[ "$CREATE_PR" =~ ^[Yy]$ ]]; then
            gh pr create --base develop --head $CURRENT_BRANCH --title "$COMMIT_MESSAGE"
        fi
    else
        # Main/develop branch flow
        git add . && \
        git commit -m "$COMMIT_MESSAGE" && \
        git push -u origin $CURRENT_BRANCH
        
        if [ "$CURRENT_BRANCH" == "develop" ]; then
            echo "Changes pushed to develop branch."
            read -p "Do you want to merge these changes to main? (y/N): " MERGE_TO_MAIN
            if [[ "$MERGE_TO_MAIN" =~ ^[Yy]$ ]]; then
                git checkout main && \
                git merge develop && \
                git push origin main && \
                git checkout develop
                echo "Changes merged and pushed to main branch. Returned to develop branch."
            fi
        else
            echo "Changes pushed to main branch."
        fi
    fi
    
    if [ $? -eq 0 ]; then
        echo "Successfully committed changes and pushed to GitHub."
    else
        echo "Error during push. Check your remote configuration and try again."
        exit 1
    fi
fi

# Show current status
echo "Current repository status:"
git status --short

# Show recent commits
echo "Recent commits:"
git log --oneline -n 5