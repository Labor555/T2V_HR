#!/usr/bin/env bash

remote_python_prefix() {
  local env_name="${REMOTE_CONDA_ENV:-t2v_hr}"
  if [[ -n "${REMOTE_PYTHON:-}" ]]; then
    echo "'${REMOTE_PYTHON}'"
  elif [[ -n "${REMOTE_CONDA_EXE:-}" ]]; then
    echo "'${REMOTE_CONDA_EXE}' run -n '${env_name}'"
  elif [[ -x "/mnt/nfs/users/lbzhu/conda/bin/conda" ]]; then
    echo "'/mnt/nfs/users/lbzhu/conda/bin/conda' run -n '${env_name}'"
  else
    echo "conda run -n '${env_name}'"
  fi
}

remote_env_prefix() {
  if [[ -n "${REMOTE_PYTHON:-}" ]]; then
    echo "PATH='$(dirname "${REMOTE_PYTHON}")':\$PATH"
  else
    echo ""
  fi
}
