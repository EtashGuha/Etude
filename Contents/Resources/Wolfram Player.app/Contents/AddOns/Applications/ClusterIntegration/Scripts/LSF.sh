#!/bin/sh

# Cluster Integration Package for gridMathematica
#
# Copyright (c) 2006 - 2007 Wolfram Research, Inc. All rights reserved.
#
# Package version: 1.3
#
# gridMathematica version: 2.2
#
# Summary: LSF remote kernel script


# Get arguments

while [ "$#" -gt "0" ]; do
  case "$1" in
    "-lsf")
       shift
       LSF_BIN_PATH="$1"
       export LSF_BIN_PATH
       shift;;

    "-lsf_opts")
       shift
       LSF_OPTIONS="$1"
       export LSF_OPTIONS
       shift;;

    "-math")
       shift
       MATH_KERNEL="$1"
       export MATH_KERNEL
       shift;;

    "-linkname")
       shift
       MATH_LINKNAME="$1"
       export MATH_LINKNAME
       shift;;

    "-help")
       echo "Usage: ";
       echo "`basename $0` -linkname <linkname> \
[-math <math_kernel_path>] [-lsf <lsf_bin_path>] [-lsf_opts <lsf_option>]"
       echo "  <linkname> has no default value and must be supplied"
       echo "  <math_kernel_path> defaults to \"/usr/local/bin/math\""
       echo "  <lsf_bin_path> defaults to \"/usr/local/lsf/bin\""
       echo "  <lsf_option> has no default value. Options must be passed as a \
quoted string."
       exit 0;;

    *)
       shift;;
  esac
done

# Set up environment & Mathematica parameters

if [ -z "${MATH_LINKNAME}" ]; then
  echo "ERROR: Mathematica LinkName not supplied. Use -linkname <linkname>"
  exit
fi

if [ -z "${LSF_BIN_PATH}" ]; then
  LSF_BIN_PATH="/usr/local/lsf/bin"
fi
if [ ! -d "${LSF_BIN_PATH}" ]; then
  echo "ERROR: ${LSF_BIN_PATH} is not a valid directory."
  exit 1
fi
if [ ! -x "${LSF_BIN_PATH}/bsub" ]; then
  echo "ERROR: Cannot find LSF in ${LSF_BIN_PATH}"
  exit 1
fi

if [ -z "${MATH_KERNEL}" ]; then
  MATH_KERNEL="/usr/local/bin/math"
fi
if [ ! -x "${MATH_KERNEL}" ]; then
  echo "ERROR: Cannot run Mathematica kernel '${MATH_KERNEL}'"
  exit 1
fi

MATH_OPTIONS="-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName '${MATH_LINKNAME}'"

LSF_COMMAND="${LSF_BIN_PATH}/bsub ${LSF_OPTIONS} ${MATH_KERNEL} ${MATH_OPTIONS}"


#Run the job

lsf_jobid_string=`eval $LSF_COMMAND`
lsf_jobid=`echo "${lsf_jobid_string}" | awk '{print $2}' | tr -d '<' | tr -d '>'`

bjobOut="RUN"
bjobPENDCount=0
bjobRUNCount=0

while [ "${bjobOut}" = "PEND" -o "${bjobOut}" = "RUN" ]; do
   sleep 1
   bjobOut=`${LSF_BIN_PATH}/bjobs "${lsf_jobid}" | sed '2p;d' | awk '{print $3}'`
   if [ "${bjobOut}" = "PEND" ]; then
     bjobPENDCount=`expr ${bjobPENDCount} + 1`
     if [ "${bjobPENDCount}" -gt "10" ]; then
         ${LSF_BIN_PATH}/bkill "${lsf_jobid}"
         exit 1
     fi
   fi
   if [ "${bjobOut}" = "RUN" ]; then
     bjobRUNCount=`expr ${bjobRUNCount} + 1`
     if [ "${bjobRUNCount}" -gt "3" ]; then
         break
     fi
   fi
done

exit 0

