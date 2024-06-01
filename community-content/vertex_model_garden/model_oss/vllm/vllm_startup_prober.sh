#!/bin/bash

# !/bin/bash
# The Startup prober is built to check whether the server is ready to
# serve traffic. The stript returns 0 if succeed. Any other returned
# value are consider as an error. More detail could be found from
# [shell script Exit codes](http://shellscript.sh/exitcodes.html).

PORT=7080
check_model_availability(){
  curl -s -o /dev/null -w "%{http_code}" "http://0.0.0.0:${PORT}/health" | grep "200" -q
}

main(){
  check_model_availability
  local available=$?
  if [[ $available -gt 0 ]]
  then
    echo "Warning: vLLM server is not yet available."
    return 1
  fi
  return 0
}
main

