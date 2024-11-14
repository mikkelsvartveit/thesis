#!/bin/bash

# =================================
# Snekrer wordcount basert på git diff fra forrige commit
# ./wordcount-diff 4 gir wordcount diff fra 4 siste commits
# =================================


# Function to get word count for a file
get_word_count() {
    local file="$1"
    wc -w "$file" | awk '{print $1}'
}

# Go to content directory
cd content || exit 1

# Find all markdown files excluding those starting with 00 or 9
files=$(find . -name "*.md" | grep -v -E "(00|9[0-9])-" | sort)

total_diff=0
changes_found=false


commits_back=${1:-0}
comparison_commit="HEAD"
if [ "$commits_back" -gt 0 ]; then
    comparison_commit="HEAD~$commits_back"
fi

echo "Word count changes in this session:"
echo "--------------------------------"

for file in $files; do
    # Check if file is tracked by git
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        # Get current word count
        current_count=$(get_word_count "$file")
        
        # Get word count from last commit
        last_committed_count=$(git show "$comparison_commit":"$file" 2>/dev/null | wc -w)
        
        # Calculate difference
        diff=$((current_count - last_committed_count))
        
        if [ $diff -ne 0 ]; then
            changes_found=true
            total_diff=$((total_diff + diff))
            if [ $diff -gt 0 ]; then
                echo -e "$file: \033[32m+$diff\033[0m words"
            else
                echo -e "$file: \033[31m$diff\033[0m words"
            fi
        fi
    else
        # New file
        changes_found=true
        current_count=$(get_word_count "$file")
        echo "$file: +$current_count words (new file)"
        total_diff=$((total_diff + current_count))
    fi
done

if [ "$changes_found" = true ]; then
    echo "--------------------------------"
    echo "Total changes: $total_diff words"
else
    echo "No changes found in tracked markdown files."
fi