import subprocess
import sys


def get_git_version():
    """Derive the version string from `git describe --tags --long`.

    Returns a plain tag (e.g. "6.0.0") when sitting exactly on a tag, or a PEP
    440 development version (e.g. "6.0.0.dev12") when there are commits since
    the last tag.
    """
    result = subprocess.run(
        ["git", "describe", "--tags", "--long"], capture_output=True, text=True
    ).stdout
    tag, nplus, commit = result.split("-")

    if "-v" in sys.argv or "--show" in sys.argv:
        print(f"Last tag: {tag}")
        print(f"Number of commits since tag: {nplus}")
        print(f"Commit object name: {commit[1:] if commit.startswith('g') else commit}")

    # Strip a leading v if it is there
    tag = f"{tag[1:] if tag.startswith('v') else tag}"

    # This should be a legal Python version format
    return f"{tag}.dev{nplus}" if int(nplus) else tag


def write_version(version):
    """Stamp `version` into meson.build, pyproject.toml, and the package __init__."""
    targets = [
        ("meson.build", "version:", f"  version: '{version}',\n"),
        ("python/pyproject.toml", "version", f"version = '{version}'\n"),
        ("python/presto/__init__.py", "__version__", f"__version__ = '{version}'\n"),
    ]
    for path, key, replacement in targets:
        with open(path, "r+") as f:
            ll = f.readlines()
            f.seek(0)
            for line in ll:
                ls = line.split()
                if len(ls) >= 2 and ls[0] == key:
                    f.write(replacement)
                    continue
                f.write(line)
            # truncate() so a shorter version string can't leave stale bytes behind
            f.truncate()


# An explicit "--set X.Y.Z" (for a tagged release) overrides the git-derived
# development version; otherwise the version comes from `git describe`.
if "--set" in sys.argv:
    version = sys.argv[sys.argv.index("--set") + 1]
    version = version[1:] if version.startswith("v") else version
else:
    version = get_git_version()

print(version)

if "-w" in sys.argv or "--write" in sys.argv:
    write_version(version)
