#!/bin/bash

usage ()
{
    cat <<EOF
USAGE: merge_csv.sh DST SRC...

Merge and xz compress multiple csv files together.
NOTE: This script will ignore rows with nan values!
EOF
    exit 1
}

die ()
{
    echo "$@"
    exit 1
}

data_generator ()
{
    local src=( "${@}" )
    head -n 1 "${src[0]}"

    for (( i = 0; i < "${#src[@]}"; i++ ))
    do
        echo -n -e "[$((i+1)) / ${#src[@]}] Processing file ${src[$i]}\r" >&2
        tail -n +2 "${src[$i]}" | grep -v nan
    done
}

dst="${1}"
shift
src=( "${@}" )

[[ "${#src[@]}" -eq 0 ]] && usage
[[ -f "${dst}" ]] && die "Destination file exist already. Will not overwrite."

data_generator "${src[@]}" | xz -9 -T 0 > "${dst}"

echo -e "\nDONE"

