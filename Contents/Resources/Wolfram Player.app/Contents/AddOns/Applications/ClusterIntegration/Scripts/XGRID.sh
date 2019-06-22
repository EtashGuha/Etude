#!/bin/sh

# Cluster Integration Package for Mathematica
#
# Copyright (c) 2006 - 2008 Wolfram Research, Inc. All rights reserved.
#
# Mathematica version: 7.0
#
# Summary: XGRID remote kernel script


# Get arguments

while [ "$#" -gt "0" ]; do
  case "$1" in
    "-xgrid")
       shift
       XGRID_BIN_PATH="$1"
       export XGRID_BIN_PATH
       shift;;

    "-xgrid_opts")
       shift
       XGRID_OPTIONS="$1"
       export XGRID_OPTIONS
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
[-math <math_kernel_path>] [-xgrid <xgrid_bin_path>] [-xgrid_opts <xgrid_option>]"
       echo "  <linkname> has no default value and must be supplied."
       echo "  <math_kernel_path> defaults to \"/Applications/Mathematica.app/Contents/MacOS/MathKernel\"."
       echo "  <xgrid_bin_path> defaults to \"/usr/bin\"."
       echo "  <xgrid_option> has no default value. Options must be passed as a \
quoted string."
       exit 0;;

    *)
       shift;;
  esac
done


# Set up environment and Mathematica parameters

if [ -z "${MATH_LINKNAME}" ]; then
  echo "ERROR: Mathematica LinkName not supplied. Use -linkname <linkname>"
  exit
fi

if [ -z "${XGRID_BIN_PATH}" ]; then
  XGRID_BIN_PATH="/usr/bin"
fi
if [ ! -d "${XGRID_BIN_PATH}" ]; then
  echo "ERROR: ${XGRID_BIN_PATH} is not a valid directory."
  exit 1
fi
if [ ! -x "${XGRID_BIN_PATH}/xgrid" ]; then
  echo "ERROR: Cannot find XGRID in ${XGRID_BIN_PATH}"
  exit 1
fi

if [ -z "${MATH_KERNEL}" ]; then
  MATH_KERNEL="/Applications/Mathematica.app/Contents/MacOS/MathKernel"
fi
if [ ! -x "${MATH_KERNEL}" ]; then
  echo "ERROR: Cannot run Mathematica kernel '${MATH_KERNEL}'"
  exit 1
fi

MATH_OPTIONS="-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName '${MATH_LINKNAME}'"

XGRID_COMMAND="${XGRID_BIN_PATH}/xgrid ${XGRID_OPTIONS} -job submit ${MATH_KERNEL} ${MATH_OPTIONS}"


# Run the job

xgrid_jobid_string=`eval $XGRID_COMMAND`
xgrid_jobid=`echo "${xgrid_jobid_string}" | awk '{print $2}' | tr -d '<' | tr -d '>'`

bjobOut="Running"
bjobPENDCount=0
bjobRUNCount=0

while [ "${bjobOut}" = "Prepared" -o "${bjobOut}" = "Running" ]; do
   sleep 1
   bjobOut=`${XGRID_BIN_PATH}/xgrid ${XGRID_OPTIONS} -job attributes -id "${xgrid_jobid}" | sed '2p;d' | awk '{print $3}'`
   if [ "${bjobOut}" = "Prepared" ]; then
     bjobPENDCount=`expr ${bjobPENDCount} + 1`
     if [ "${bjobPENDCount}" -gt "10" ]; then
         ${XGRID_BIN_PATH}/xgrid ${XGRID_OPTIONS} -job delete -id "${xgrid_jobid}"
         exit 1
     fi
   fi
   if [ "${bjobOut}" = "Running" ]; then
     bjobRUNCount=`expr ${bjobRUNCount} + 1`
     if [ "${bjobRUNCount}" -gt "3" ]; then
         break
     fi
   fi
done

exit 0

s