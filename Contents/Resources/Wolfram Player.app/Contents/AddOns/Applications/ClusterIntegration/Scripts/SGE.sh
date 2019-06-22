#!/bin/sh
#
#
##########################################################################
#
#
#  Copyright (c) 2006 Wolfram Research, Inc.
#
#
##########################################################################

# Initialization

#$ -S /bin/sh
#$ -N Wolfram Mathematica

# Redirect input and output to /dev/null

#$ -e /dev/null
#$ -o /dev/null

# Get arguments

if [ $# -eq 2 ]; then
   MATH_BINARY=$1
   MATH_LINKNAME=$2
else
   exit 1
fi

# Parameters for Mathematica Kernel

MATH_OPTIONS="-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName $MATH_LINKNAME"

$MATH_BINARY $MATH_OPTIONS


