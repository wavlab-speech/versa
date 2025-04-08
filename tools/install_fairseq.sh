#!/bin/bash

# Exit on any error
set -e

# Repository information
REPO_OWNER="ftshijt"
REPO_NAME="fairseq"
REPO_PATH="$REPO_OWNER/$REPO_NAME"
BRANCH="versa"
EXPECTED_COMMIT_ID="a3fdd28642d218789265bd42f2411a20ba892538"

# Function to check if repository exists
check_repo_exists() {
    local repo_path=$1
    echo "Verifying repository exists: $repo_path"
    
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://github.com/$repo_path")
    
    if [ $HTTP_STATUS -eq 404 ]; then
        echo "ERROR: Repository not found: $repo_path"
        exit 1
    elif [ $HTTP_STATUS -ne 200 ]; then
        echo "ERROR: Could not access GitHub. HTTP Status: $HTTP_STATUS"
        exit 1
    fi
    
    echo "Repository verification successful: $repo_path exists"
}

# Function to check if branch exists
check_branch_exists() {
    local repo_path=$1
    local branch=$2
    echo "Verifying branch exists: $branch"
    
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://github.com/$repo_path/tree/$branch")
    
    if [ $HTTP_STATUS -eq 404 ]; then
        echo "ERROR: Branch not found: $branch in repository $repo_path"
        exit 1
    elif [ $HTTP_STATUS -ne 200 ]; then
        echo "ERROR: Could not access GitHub branch. HTTP Status: $HTTP_STATUS"
        exit 1
    fi
    
    echo "Branch verification successful: $branch exists in repository $repo_path"
}

# Function to check if commit exists (if a specific commit is required)
check_commit_exists() {
    local repo_path=$1
    local commit_id=$2
    
    if [ -z "$commit_id" ]; then
        echo "No specific commit ID provided, skipping commit verification"
        return 0
    fi
    
    echo "Verifying commit exists: $commit_id"
    
    # Validate commit ID format
    if [[ ! $commit_id =~ ^[a-fA-F0-9]+$ ]]; then
        echo "ERROR: Invalid commit ID format: $commit_id (should be hexadecimal)"
        exit 1
    fi
    
    COMMIT_URL="https://github.com/$repo_path/commit/$commit_id"
    COMMIT_HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$COMMIT_URL")
    
    if [ $COMMIT_HTTP_STATUS -eq 404 ]; then
        echo "ERROR: Commit not found: $commit_id in repository $repo_path"
        exit 1
    elif [ $COMMIT_HTTP_STATUS -ne 200 ]; then
        echo "ERROR: Could not access GitHub commit. HTTP Status: $COMMIT_HTTP_STATUS"
        exit 1
    fi
    
    echo "Commit verification successful: $commit_id exists in repository $repo_path"
}

# Function to check if local copy is at the expected commit
check_local_commit() {
    local expected_commit=$1
    
    if [ -z "$expected_commit" ]; then
        return 1 # No commit specified, so we should reinstall
    fi
    
    if [ ! -d "fairseq" ]; then
        return 1 # Directory doesn't exist, so we should install
    fi
    
    # Change to the fairseq directory to check git status
    cd fairseq
    
    # Check if this is a git repository
    if [ ! -d ".git" ]; then
        cd ..
        return 1 # Not a git repository, so we should reinstall
    fi
    
    # Check if repo is on the right branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
        echo "Branch mismatch! Current: $CURRENT_BRANCH, Expected: $BRANCH"
        cd ..
        return 1 # Wrong branch, so we should reinstall
    fi
    
    # Get the current commit hash
    CURRENT_COMMIT=$(git rev-parse HEAD)
    SHORT_CURRENT_COMMIT=${CURRENT_COMMIT:0:${#expected_commit}}
    
    cd .. # Return to the original directory
    
    # Check if the commit matches
    if [[ "$SHORT_CURRENT_COMMIT" != "$expected_commit"* ]]; then
        echo "Commit mismatch! Current: $CURRENT_COMMIT, Expected: $expected_commit"
        return 1 # Wrong commit, so we should reinstall
    fi
    
    # If we get here, the local copy is at the expected commit
    return 0
}

# Function to validate the specific commit after cloning (if a specific commit is required)
validate_cloned_commit() {
    local expected_commit=$1
    
    if [ -z "$expected_commit" ]; then
        return 0
    fi
    
    echo "Validating cloned repository commit..."
    
    # Get the current commit hash
    CURRENT_COMMIT=$(git rev-parse HEAD)
    SHORT_CURRENT_COMMIT=${CURRENT_COMMIT:0:${#expected_commit}}
    
    if [[ "$SHORT_CURRENT_COMMIT" != "$expected_commit"* ]]; then
        echo "ERROR: Commit mismatch!"
        echo "Expected commit starting with: $expected_commit"
        echo "Actual commit: $CURRENT_COMMIT"
        exit 1
    fi
    
    echo "Commit validation successful: Repository is at expected commit"
}

echo "=== Starting fairseq installation with validation ==="

# Verify repository, branch, and commit exist before proceeding
check_repo_exists "$REPO_PATH"
check_branch_exists "$REPO_PATH" "$BRANCH"

if [ -n "$EXPECTED_COMMIT_ID" ]; then
    check_commit_exists "$REPO_PATH" "$EXPECTED_COMMIT_ID"
fi

# Check if we already have the correct version installed
if [ -n "$EXPECTED_COMMIT_ID" ] && check_local_commit "$EXPECTED_COMMIT_ID"; then
    echo "Local repository is already at the expected commit: $EXPECTED_COMMIT_ID"
    echo "Skipping reinstallation."
    exit 0
fi

# Clean up existing directory if it exists
if [ -d "fairseq" ]; then
    echo "Removing existing fairseq directory..."
    rm -rf fairseq
fi

# Clone the repository
echo "Cloning repository: $REPO_PATH (branch: $BRANCH)..."
git clone -b "$BRANCH" "https://github.com/$REPO_PATH.git"
cd fairseq

# Validate the commit if specified
if [ -n "$EXPECTED_COMMIT_ID" ]; then
    validate_cloned_commit "$EXPECTED_COMMIT_ID"
fi

# Install the package
echo "Installing fairseq package..."
pip install -e .
cd ..

echo "=== fairseq installation completed successfully ==="
