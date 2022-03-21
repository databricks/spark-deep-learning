#!/bin/bash

_bsd_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec prospector --profile ${_bsd_}/prospector.yaml ${_bsd_}/python
