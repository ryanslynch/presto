import subprocess
import sys

# Determine the version info from git
result = subprocess.run(
    ["git", "describe", "--tags", "--long"], capture_output=True, text=True
).stdout
tag, nplus, commit = result.split("-")

if len(sys.argv) > 1 and sys.argv[1] in ["-v", "--show"]:
    print(f"Last tag: {tag}")
    print(f"Number of commits since tag: {nplus}")
    print(f"Commit object name: {commit[1:] if commit.startswith('g') else commit}")

# Strip a leading v if it is there
tag = f"{tag[1:] if tag.startswith('v') else tag}"

# This should be a legal Python version format
version = f"{tag}.dev{nplus}" if int(nplus) else tag
print(version)

if len(sys.argv) > 1 and sys.argv[1] in ["-w", "--write"]:
    # Update the version info in meson.build and the python pyproject.toml files
    with open("meson.build", "r+") as f:
        ll = f.readlines()
        f.seek(0)
        for line in ll:
            ls = line.split()
            if len(ls) >= 2:
                if ls[0]=="version:":
                    f.write(f"  version: '{version}',\n")
                    continue
            f.write(line)
        f.truncate()
    # Update the version info in the python pyproject.toml file
    with open("python/pyproject.toml", "r+") as f:
        ll = f.readlines()
        f.seek(0)
        for line in ll:
            ls = line.split()
            if len(ls) >= 2:
                if ls[0]=="version":
                    f.write(f"version = '{version}'\n")
                    continue
            f.write(line)
        f.truncate()
    # Update the version info in the presto package __init__.py
    with open("python/presto/__init__.py", "r+") as f:
        ll = f.readlines()
        f.seek(0)
        for line in ll:
            ls = line.split()
            if len(ls) >= 2:
                if ls[0]=="__version__":
                    f.write(f"__version__ = '{version}'\n")
                    continue
            f.write(line)
        f.truncate()
