This directory contains strong-named versions of the .NET/Link assemblies.

The standard assemblies are not strong-named. This is done for several reasons,
among them the fact that it makes it possible for user-written .NET apps that
bundle the Wolfram.NETLink.dll assembly to be upgraded with newer versions
of that assembly without being recompiled.

In some cases, however, programmers will want to call the Wolfram.NETLink.dll
assembly from their own strong-named assembly. In that case, the
Wolfram.NETLink.dll assembly must also be strong-named (because strong-named
assemblies can only call other strong-named assemblies). If you need a
strong-named version of Wolfram.NETLink.dll for your application, you can
use the one in this directory.

This issue is generally not relevant for users who just want to call
.NET from Mathematica, not vice-versa. In some rare circumstances, however,
you might need to make sure that the .NET/Link assemblies used by the
InstallNET[] function in Mathematica are the strong-named versions. In
this case, you can replace both the InstallableNET.exe and Wolfram.NETLink.dll
files from the root NETLink directory with the strong-named versions from
this directory. Note that you cannot keep the old InstallableNET.exe file
and just swap in the strong-named Wolfram.NETLink.dll--you need to use 
both strong-named versions together.