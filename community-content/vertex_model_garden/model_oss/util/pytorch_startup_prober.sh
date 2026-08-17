#!/bin/bash

# !/bin/bash
# The Startup prober built to check whether models listed in local disk are
# loaded in memory and are ready to serve traffic. The script returns 0 if
# succeed. Any other returned value are consider as an error. More detail could be
# found from [shell script Exit codes](http://shellscript.sh/exitcodes.html).
#
# TorchServe: The Management API listens on port 8081 and is only accessible
# from localhost by default.

if [[ -z "${MNG_PORT}" ]]; then
  MNG_PORT=7081  # We default the management_port to 7081.
else
  MNG_PORT="${MNG_PORT}"
fi

check_model_availability(){
  local MODEL_NAME=$1
  # Returns whether "READY" is found in the model status.
  # Reference: https://pytorch.org/serve/management_api.html#describe-model.
  curl -s "http://localhost:${MNG_PORT}/models/${MODEL_NAME}" | grep "READY" -q
}

main(){
  check_model_availability "$MODEL"  # Assume Dockerfile sets MODEL environment parameter.
  local available=$?
  if [[ $available -gt 0 ]]
  then
    echo "Warning: Model(${MODEL}) is not yet available."
    return 1
  fi
  return 0
}
main
