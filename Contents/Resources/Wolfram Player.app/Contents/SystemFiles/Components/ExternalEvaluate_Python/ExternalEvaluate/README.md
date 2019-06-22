# ExternalEvaluate

* ExternalEvaluatePython.wl
 * This file registers the Python system for ExternalEvaluate (See ExternalEvaluate/Kernel/Main.m)
 * This also registers Python-PIL and Python-NumPy
 * ExportPythonExpression, this file contains code to transform Wolfram Language expressions to Python expressions.
 * ImportPythonExpression, this file contains code to evaluate and import Python expressions back into Wolfram Language.

* Python evaluations are launched using the code in ExternalEvaluate_Python/Resources/wolframevaluate
* Further serialization code relies on https://stash.wolfram.com/projects/LCL/repos/wolframclientforpython/browse/wolframclient