#! /bin/sh
#
#   Copyright (c) 2003 Wolfram Research, Inc.  All rights reserved.
#  
#   Sample start script for an GUIKitApplication
#
#   Make sure MATH_HOME and GUIKIT_HOME below
#   and optionally JAVA_HOME are defined correctly
#   for your configuration or set them in your environment
#
#   NOTE: you must pass GUIKit` definition relative names to this script 
#    for it to actually run anything.  See the cooresponding PrimeFinderApplication.command
#    script for an example of a specific user interface definition launch


# OS specific support.
darwin=false;
case "`uname`" in
	Darwin*) darwin=true
		if [ -z "$JAVA_HOME" ] ; then
			JAVA_HOME=/System/Library/Frameworks/JavaVM.framework/Home   
		fi
		;;
esac

#   Change this path to where Mathematica is located or define this in your environment
if [ -z "$MATH_HOME" ] ; then 
	if $darwin ; then
			MATH_HOME="/Applications/Mathematica 5.0.app"
	else
  		MATH_HOME="/usr/local/Wolfram/Mathematica/5.0"
	fi
fi

if [ -z "$MATH_COMMANDLINE" ] ; then 
	if $darwin ; then
			MATH_COMMANDLINE="-linkmode launch -linkname '\"${MATH_HOME}/Contents/MacOS/MathKernel\" -mathlink'"
	else
  		MATH_COMMANDLINE="-linkmode launch -linkname '${MATH_HOME}/Executables/math -mathlink'"
	fi
fi

echo "Using Mathematica from $MATH_HOME"

#   Change this path to where you have installed the GUIKit` AddOn or define this in your environment
if [ -z "$GUIKIT_HOME" ] ; then 
	GUIKIT_HOME="${MATH_HOME}/AddOns/Applications/GUIKit/"
fi

echo "Using GUIKit from $GUIKIT_HOME"

LOCALCLASSPATH="${MATH_HOME}/AddOns/JLink/JLink.jar"
# add in the dependency .jar files in non-RPM mode (the default)
for i in "${GUIKIT_HOME}/Java"/*.jar
do
  # if the directory is empty, then it will return the input string
  # this is stupid, so case for it
  if [ -f "$i" ] ; then
    if [ -z "$LOCALCLASSPATH" ] ; then
      LOCALCLASSPATH="$i"
    else
      LOCALCLASSPATH="$i":"$LOCALCLASSPATH"
    fi
  fi
done

if [ -z "$JAVACMD" ] ; then
  if [ -n "$JAVA_HOME"  ] ; then
    if [ -x "$JAVA_HOME/jre/sh/java" ] ; then
      # IBM's JDK on AIX uses strange locations for the executables
      JAVACMD="$JAVA_HOME/jre/sh/java"
    else
      JAVACMD="$JAVA_HOME/bin/java"
    fi
	else
		JAVACMD=`which java 2> /dev/null `
		if [ -z "$JAVACMD" ] ; then
			JAVACMD=java
		fi
  fi
fi

echo "Using Java VM from $JAVA_HOME"

"$JAVACMD" $JAVA_OPTS -cp "$LOCALCLASSPATH" -Dbsf_engine_Mathematica_KernelLinkCommandLine="$MATH_COMMANDLINE" com.wolfram.guikit.app.GUIKitApplication "$@"
