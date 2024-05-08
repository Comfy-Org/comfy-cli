#!/bin/bash

# Get a list of changed files in the current branch compared to the main branch
FILES=$(git diff --name-only main | grep '\.py$')

# Run Black on each Python file
echo "Formatting changed Python files..."
for file in $FILES; do
    black $file
done

echo "Formatting complete."
