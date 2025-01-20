#!/bin/bash

# =================================
# Snekrer wordcount basert på git diff fra forrige commit
# ./wordcount-diff 4 gir wordcount diff fra 4 siste commits
# =================================


# Function to get word count for a file
get_word_count() {
    local file="$1"
    #wc -w "$file" | awk '{print $1}'
    python3 -c "import re; print(len(re.sub(r'<!--[\s\S]*?-->', '', open('$file').read()).split()))"
}

# Go to content directory
cd content || exit 1

# Find all markdown files excluding those starting with 00 or 9
files=$(find . -name "*.md" | grep -v -E "(00|9[0-9])-" | sort)

total=0
total_diff=0
changes_found=false


commits_back=${1:-0}
comparison_commit="HEAD"
if [ "$commits_back" -gt 0 ]; then
    comparison_commit="HEAD~$commits_back"
fi

for file in $files; do
    # Check if file is tracked by git
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
        # Get current word count
        current_count=$(get_word_count "$file" )
        
        # Get word count from last commit
        last_committed_count=$(get_word_count <(git show "$comparison_commit":"$file" 2>/dev/null))
        
        # Calculate difference
        diff=$((current_count - last_committed_count))
        total=$(($total + $current_count))
        
        printf "%7d %-20s" "$current_count" "$file"
        if [ $diff -ne 0 ]; then
            changes_found=true
            total_diff=$((total_diff + diff))
            if [ $diff -gt 0 ]; then
                echo -e " (\033[32m+$diff\033[0m)"
            else
                echo -e " (\033[31m$diff\033[0m)"
            fi
        else
            echo
        fi
    else
        # New file
        changes_found=true
        current_count=$(get_word_count "$file")
        echo "$file: +$current_count words (new file)"
        total_diff=$((total_diff + current_count))
    fi
done

echo "--------------------------------"
printf "%7d total\n" "$total"
printf "%7d diff\n" "$total_diff"

