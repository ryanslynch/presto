#!/bin/sh
# Convert PRESTO prepfold .ps plots into nice anti-aliased .png images.
#
# This used to call latex2html's 'pstoimg', but that package is not available
# on conda-forge.  pstoimg was only a wrapper that used Ghostscript ('gs') for
# the actual anti-aliased rendering, and gs is well-maintained and *is* on
# conda-forge, so we now call gs directly.  The -dTextAlphaBits/-dGraphicsAlphaBits
# flags give the same anti-aliasing pstoimg used, and Orientation 3 reproduces
# pstoimg's "-flip cw".
#
# Original latex2html/pstoimg version, kept for reference:
#   pstoimg -density 200 -antialias -type png -flip cw $@
for ps in "$@"; do
    png=`echo "$ps" | sed 's/\.[^.]*$/.png/'`
    gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r200 \
       -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
       -sOutputFile="$png" -c "<</Orientation 3>> setpagedevice" -f "$ps"
done
