#!/usr/bin/env python3
"""Remove code blocks from markdown files."""

import re
import sys

def remove_code_blocks(content):
    """Remove code blocks from markdown content."""
    # Remove fenced code blocks (``` or ```)
    content = re.sub(r'```[\s\S]*?```', '', content, flags=re.MULTILINE)
    
    # Remove inline code that spans multiple lines
    content = re.sub(r'`[^`\n]+`', lambda m: m.group(0) if '\n' not in m.group(0) else '', content)
    
    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content

def process_file(filepath):
    """Process a single markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned_content = remove_code_blocks(content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"Processed {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_code_blocks.py <file1.md> [file2.md ...]")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        process_file(filepath)