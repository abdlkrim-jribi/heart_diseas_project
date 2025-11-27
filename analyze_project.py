#!/usr/bin/env python3
"""
Project Structure Analyzer
Generates a comprehensive project tree with file contents for AI context sharing
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Directories and file patterns to exclude
EXCLUDE_DIRS = {
    'venv', 'env', '__pycache__', '.git', 'node_modules',
    '.idea', '.vscode', 'dist', 'build', '.next', 'coverage',
    '.pytest_cache', '.mypy_cache', 'htmlcov'
}

EXCLUDE_FILES = {
    '.DS_Store', 'Thumbs.db', '.gitkeep', '*.pyc', '*.pyo',
    '*.egg-info', '*.keras', '*.pkl', '*.h5', '*.weights'
}

# File extensions to include content for
TEXT_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.md', '.txt',
    '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env.example',
    '.gitignore', '.dockerignore', 'Dockerfile', '.html', '.css', '.scss',
    '.sh', '.bash', '.sql', '.xml'
}

BINARY_EXTENSIONS = {'.keras', '.pkl', '.h5', '.weights', '.jpg', '.png', '.gif', '.pdf'}

def should_exclude_dir(dir_name):
    """Check if directory should be excluded"""
    return dir_name in EXCLUDE_DIRS or dir_name.startswith('.')

def should_exclude_file(file_name):
    """Check if file should be excluded"""
    if file_name in EXCLUDE_FILES:
        return True
    for pattern in EXCLUDE_FILES:
        if pattern.startswith('*') and file_name.endswith(pattern[1:]):
            return True
    return False

def get_file_size(file_path):
    """Get human-readable file size"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def should_include_content(file_path):
    """Determine if file content should be included"""
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name

    # Check if it's a binary file
    if ext in BINARY_EXTENSIONS:
        return False

    # Check if it's a text file we want to include
    if ext in TEXT_EXTENSIONS or name in ['Dockerfile', 'Makefile', '.env.example']:
        # Skip large files
        if os.path.getsize(file_path) > 500000:  # 500KB limit
            return False
        return True

    return False

def read_file_content(file_path, max_lines=1000):
    """Read file content with encoding fallback"""
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    return ''.join(lines[:max_lines]) + f"\n\n... [File truncated - showing first {max_lines} lines only] ...\n"
                return ''.join(lines)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            return f"[Error reading file: {str(e)}]"

    return "[Binary file or unsupported encoding]"

def generate_tree_structure(root_path, prefix="", output_lines=None):
    """Generate tree-style directory structure"""
    if output_lines is None:
        output_lines = []

    try:
        items = sorted(os.listdir(root_path))
    except PermissionError:
        return output_lines

    # Separate directories and files
    dirs = [item for item in items if os.path.isdir(os.path.join(root_path, item))
            and not should_exclude_dir(item)]
    files = [item for item in items if os.path.isfile(os.path.join(root_path, item))
             and not should_exclude_file(item)]

    all_items = dirs + files

    for i, item in enumerate(all_items):
        is_last = i == len(all_items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "

        item_path = os.path.join(root_path, item)

        if os.path.isdir(item_path):
            output_lines.append(f"{prefix}{current_prefix}{item}/")
            generate_tree_structure(item_path, prefix + next_prefix, output_lines)
        else:
            size = get_file_size(item_path)
            output_lines.append(f"{prefix}{current_prefix}{item} ({size})")

    return output_lines

def generate_detailed_structure(root_path, base_path=None, indent=0):
    """Generate detailed structure with file contents"""
    if base_path is None:
        base_path = root_path

    output = []
    indent_str = "  " * indent

    try:
        items = sorted(os.listdir(root_path))
    except PermissionError:
        return output

    # Separate directories and files
    dirs = [item for item in items if os.path.isdir(os.path.join(root_path, item))
            and not should_exclude_dir(item)]
    files = [item for item in items if os.path.isfile(os.path.join(root_path, item))
             and not should_exclude_file(item)]

    # Process directories first
    for dir_name in dirs:
        dir_path = os.path.join(root_path, dir_name)
        rel_path = os.path.relpath(dir_path, base_path)

        output.append(f"\n{indent_str}{'='*80}")
        output.append(f"{indent_str}📁 DIRECTORY: {rel_path}/")
        output.append(f"{indent_str}{'='*80}\n")

        output.extend(generate_detailed_structure(dir_path, base_path, indent + 1))

    # Process files
    for file_name in files:
        file_path = os.path.join(root_path, file_name)
        rel_path = os.path.relpath(file_path, base_path)
        size = get_file_size(file_path)

        output.append(f"\n{indent_str}{'-'*80}")
        output.append(f"{indent_str}📄 FILE: {rel_path}")
        output.append(f"{indent_str}Size: {size}")
        output.append(f"{indent_str}{'-'*80}")

        if should_include_content(file_path):
            content = read_file_content(file_path)
            output.append(f"{indent_str}Content:")
            output.append(f"{indent_str}```")
            # Indent file content
            for line in content.split('\n'):
                output.append(f"{indent_str}{line}")
            output.append(f"{indent_str}```\n")
        else:
            ext = Path(file_path).suffix.lower()
            if ext in BINARY_EXTENSIONS:
                output.append(f"{indent_str}[Binary file - content not included]\n")
            else:
                output.append(f"{indent_str}[Large file or excluded extension - content not included]\n")

    return output

def count_project_stats(root_path):
    """Count project statistics"""
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'total_size': 0,
        'file_types': {},
        'largest_files': []
    }

    for root, dirs, files in os.walk(root_path):
        # Filter directories
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]

        stats['total_dirs'] += len(dirs)

        for file in files:
            if should_exclude_file(file):
                continue

            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                stats['total_files'] += 1
                stats['total_size'] += size

                ext = Path(file).suffix.lower() or 'no_extension'
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1

                stats['largest_files'].append((file_path, size))
            except:
                pass

    # Keep only top 10 largest files
    stats['largest_files'] = sorted(stats['largest_files'], key=lambda x: x[1], reverse=True)[:10]

    return stats

def format_size(size):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def main():
    """Main function to generate project structure"""
    project_root = Path(__file__).parent
    output_file = project_root / "PROJECT_STRUCTURE.txt"

    print("🔍 Analyzing project structure...")
    print(f"📂 Root directory: {project_root}")

    # Collect statistics
    stats = count_project_stats(project_root)

    # Generate output
    output_lines = []

    # Header
    output_lines.append("="*100)
    output_lines.append("PROJECT STRUCTURE ANALYSIS")
    output_lines.append("="*100)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Project Root: {project_root}")
    output_lines.append("="*100)
    output_lines.append("")

    # Statistics
    output_lines.append("📊 PROJECT STATISTICS")
    output_lines.append("-"*100)
    output_lines.append(f"Total Directories: {stats['total_dirs']}")
    output_lines.append(f"Total Files: {stats['total_files']}")
    output_lines.append(f"Total Size: {format_size(stats['total_size'])}")
    output_lines.append("")
    output_lines.append("File Types Distribution:")
    for ext, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
        output_lines.append(f"  {ext}: {count} files")
    output_lines.append("")
    output_lines.append("Top 10 Largest Files:")
    for file_path, size in stats['largest_files']:
        rel_path = os.path.relpath(file_path, project_root)
        output_lines.append(f"  {rel_path}: {format_size(size)}")
    output_lines.append("")
    output_lines.append("="*100)
    output_lines.append("")

    # Tree structure
    output_lines.append("🌳 TREE STRUCTURE")
    output_lines.append("-"*100)
    output_lines.append(f"{project_root.name}/")
    tree_lines = generate_tree_structure(project_root)
    output_lines.extend(tree_lines)
    output_lines.append("")
    output_lines.append("="*100)
    output_lines.append("")

    # Detailed structure with file contents
    output_lines.append("📑 DETAILED FILE CONTENTS")
    output_lines.append("="*100)
    detailed_lines = generate_detailed_structure(project_root)
    output_lines.extend(detailed_lines)

    # Footer
    output_lines.append("\n" + "="*100)
    output_lines.append("END OF PROJECT STRUCTURE")
    output_lines.append("="*100)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"✅ Project structure saved to: {output_file}")
    print(f"📄 Total lines: {len(output_lines)}")
    print(f"💾 File size: {format_size(os.path.getsize(output_file))}")
    print("")
    print("🤖 You can now share this file with AI assistants for complete project context!")

if __name__ == "__main__":
    main()