# Buttons.m

Certain utility functions used for `ProgressLogger` (see Loggers.m)

# Formatting.m

Certain formatting utility functions, mostly used for Testing Notebooks (in Notebooks.m).

# Loggers.m

Code for the built-in MUnit loggers: `PrintLogger` (default), `ProgressLogger`, `NotebookLogger`, `BatchPrintLogger` and `VerbosePrintLogger`.

Note that automated TestHub test runs use a custom logger (`TestLogger`) maintained elsewhere:

https://stash.wolfram.com/projects/QA/repos/testlogger/browse

Loggers are specified as an option to `TestRun`: 

```
TestRun[source, Loggers:> {logger1, logger2, ...}]
```

Example usage:

### PrintLogger

```
In[1]:= TestRun[{Test[1 + 1, 2]}, Loggers :> {PrintLogger[]}]
```
```
During evaluation of In[1]:=
Starting test run "Automatic"
.
Tests run: 1
Failures: 0
Messages Failures: 0
Skipped Tests: 0
Errors: 0
Fatal: False
```
```
Out[1]= True
```

### VerbosePrintLogger

```
In[2]:= TestRun[{Test[1 + 1, 2], Test[1, 4]}, Loggers :> {VerbosePrintLogger[]}]
```
```
During evaluation of In[2]:=
Starting test run "Automatic"
.!
Test number 2 with TestID None had a failure.
	Input: HoldForm[1]
	Expected output: HoldForm[4]
	Actual output: HoldForm[1]

Tests run: 2
Failures: 1
Messages Failures: 0
Skipped Tests: 0
Errors: 0
Fatal: False
```
```
Out[2]= False
```

# Messages.m

Code for how MUnit intercepts and processes messages thrown during test runs. These are invoked via the internal kernel event handler mechanism. Some documentation for how that works is here:

https://stash.wolfram.com/projects/KERN/repos/kernel/browse/Source/System/eventhandler.mc#32-81

# Notebooks.m

Code for Testing Notebooks. Parses cells to convert to MUnit tests and update the cells with results.

# Palette.m

More code related to Testing Notebook, mostly to do with the various buttons and menus.

# Test.m

The implementation of ``MUnit`Test``, which is the backbone of the testing framework.

# TestRun.m

Implementation of ``MUnit`TestRun``. This parses and runs .wlt/.mt files and logs the results (see also Loggers.m).

# VerificationTest.m

User-level functions implemented in this file:

* [VerificationTest](http://reference.wolfram.com/language/ref/VerificationTest.html) is based on ``MUnit`Test``.
* [TestReport](http://reference.wolfram.com/language/ref/TestReport.html) is based on ``MUnit`TestRun``.


# WRI.m

Implementations of various "specialty" tests such as `NTest`, `OrTest`, `ExactTest` etc.

### NTest 

Compares numerical results to some accuracy or precision:

```
In[1]:= NTest[1., 1.01, AccuracyGoal -> #]&/@{2, 3}

Out[1]= {-TestResult[Success]-, -TestResult[Failure]-}

In[2]:= NTest[1.*^10, 1.01*^10, PrecisionGoal -> #]&/@{2, 3}

Out[2]= {-TestResult[Success]-, -TestResult[Failure]-}
```

### ExactTest

`ExactTest[input, output]` checks if the _evaluated_ form of `input` matches the _unevaluated_ form of `output`. 

```
In[1]:= ExactTest[f[], f[]]

Out[1]= -TestResult[Success]-

In[2]:= g[]:= "hello";

In[3]:= ExactTest[g[], g[]]

Out[3]= -TestResult[Failure]-
```

### OrTest 

Compares against a list of possible outputs:

```
In[1]:= OrTest[RandomInteger[2], {0, 1, 2}]

Out[1]= -TestResult[Success]-
```

### OrNTest

Compares against a list of possible numeric results to some accuracy or precision:

```
In[1]:= OrNTest[RandomInteger[3] + 0.01, {0, 1, 2, 3}, AccuracyGoal -> 2]

Out[1]= -TestResult[Success]-
```
