#!/bin/bash

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 1 ]]; then
	target_files=(${@})
else
	target_files=($(git diff --name-only upstream/master HEAD))
fi

echo "${target_files[@]}"
pushd "${_bsd_}"
exec prospector --profile ${_bsd_}/prospector.yaml "${target_files[@]}"
popd
