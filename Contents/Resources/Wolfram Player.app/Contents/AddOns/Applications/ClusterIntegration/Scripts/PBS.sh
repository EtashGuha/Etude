#!/bin/sh

# Cluster Integration Package for gridMathematica
#
# Copyright (c) 2006 - 2007 Wolfram Research, Inc. All rights reserved.
#
# Package version: 1.3
#
# gridMathematica version: 2.2
#
# Summary: PBS Pro remote kernel script


# Get arguments

while [ "$#" -gt "0" ]; do
  case "$1" in
    "-pbs")
       shift
       PBS_BIN_PATH="$1"
       export PBS_BIN_PATH
       shift;;

    "-pbs_opts")
       shift
       PBS_OPTIONS="$1"
       export PBS_OPTIONS
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

    "-path")
       shift
       CIP_PATH="$1"
       export CIP_PATH
       shift;;

    "-help")
       echo "Usage: ";
       echo "`basename $0` -linkname <linkname> \
[-math <math_kernel_path>] [-pbs <pbs_bin_path>] [-pbs_opts <pbs_option>]"
       echo "  <linkname> has no default value and must be supplied"
       echo "  <math_kernel_path> defaults to \"/usr/local/bin/math\""
       echo "  <pbs_bin_path> defaults to \"/usr/local/pbs/bin\""
       echo "  <pbs_option> has no default value. Options must be passed as a \
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

if [ -z "${PBS_BIN_PATH}" ]; then
  PBS_BIN_PATH="/usr/pbs/bin"
fi
if [ ! -d "${PBS_BIN_PATH}" ]; then
  echo "ERROR: ${PBS_BIN_PATH} is not a valid directory."
  exit 1
fi
if [ ! -x "${PBS_BIN_PATH}/qsub" ]; then
  echo "ERROR: Cannot find PBS in ${PBS_BIN_PATH}"
  exit 1
fi

if [ -z "${MATH_KERNEL}" ]; then
  MATH_KERNEL="/usr/local/bin/math"
  export MATH_KERNEL
fi
if [ ! -x "${MATH_KERNEL}" ]; then
  echo "ERROR: Cannot run Mathematica kernel '${MATH_KERNEL}'"
  exit 1
fi

if [ -z "${CIP_PATH}" ]; then
  CIP_PATH=`dirname $0`
fi
if [ ! -d "${CIP_PATH}" ]; then
  echo "ERROR: ${CIP_PATH} is not a valid directory."
  exit 1
fi

MATH_OPTIONS="-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName ${MATH_LINKNAME}"
export MATH_OPTIONS

PBS_COMMAND="${PBS_BIN_PATH}/qsub ${PBS_OPTIONS} -V ${CIP_PATH}/CIPPBSKernel.pbs"

# Run the job

pbs_jobid=`eval $PBS_COMMAND`

# Check status

bjobOut="R"
bjobPENDCount=0
bjobRUNCount=0

while [ "${bjobOut}" = "Q" -o "${bjobOut}" = "R" ]; do
   sleep 1
   bjobOut=`${PBS_BIN_PATH}/qstat "${pbs_jobid}" | sed '3p;d' | awk '{print $5}'`
   if [ "${bjobOut}" = "Q" ]; then
     bjobPENDCount=`expr ${bjobPENDCount} + 1`
     if [ "${bjobPENDCount}" -gt "10" ]; then
         ${PBS_BIN_PATH}/qdel "${pbs_jobid}"
         exit 1
     fi
   fi
   if [ "${bjobOut}" = "R" ]; then
     bjobRUNCount=`expr ${bjobRUNCount} + 1`
     if [ "${bjobRUNCount}" -gt "3" ]; then
         break
     fi
   fi
done

exit 0

