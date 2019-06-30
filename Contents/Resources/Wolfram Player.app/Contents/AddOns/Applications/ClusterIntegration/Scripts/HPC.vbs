'******************************************************************************
'
' Cluster Integration Package for gridMathematica
'
' Author: Charles Pooh
'
' Copyright (c) 2006 - 2008 Wolfram Research, Inc. All rights reserved.
'
' Package version: 2.0
'
' Mathematica version: 7.0
'
' Summary: Microsoft High Performance Computing Server 2008 remote kernel script
'
'******************************************************************************


' Verify Arguments.Count
'

if (WScript.Arguments.Count <> 4 ) then

   msg = MsgBox ("Remote command called with " & WScript.Arguments.Count & _
          " arguments; 4 arguments expected for Kernel configuration" & _
          " options." & chr(13) & "Please refer to the section Working with" & _
          " gridMathematica Remote Kernels in the documentation.", 0, "Wolfram Mathematica")

   WScript.Quit(-1)

end if


' Definition of Shell object
'

dim WshShell

Set WshShell = CreateObject("WScript.Shell")


' Get the schedulerName and linkname
'

Set arguments = WScript.Arguments

if (arguments.Count = 4) then

   mathKernel = Chr(34) & arguments(0) & Chr(34)

   kernelOptions = arguments(1)

   linkName = arguments(2)

   schedulerName = arguments(3)

end if


' Connect to the cluster
'

Set scheduler = CreateObject("Microsoft.Hpc.Scheduler.Scheduler")

scheduler.Connect(schedulerName)


' Create and submit job
'

Set job = scheduler.CreateJob

job.Name = "Wolfram Mathematica - " & Date & " - " & Time

job.MaximumNumberOfCores = 1

job.MinimumNumberOfCores = 1

scheduler.AddJob((job))

Set task = job.CreateTask

task.Name = "MathkernelLinkedToFE"

task.CommandLine = mathKernel & " " & kernelOptions & " " & linkName

task.MaximumNumberOfCores = 1

task.MinimumNumberOfCores = 1

job.AddTask((task))

'Dim window As IntPtr

'scheduler.SetInterfaceMode False, window

scheduler.SubmitJob job, "", ""


'Check status of job
'

WScript.Sleep 5000

job.Refresh()

Set status = job.State.ToString

if (status = "Queued") then

   Set counter = scheduler.ClusterCounter

   msg = MsgBox (counter.NumberOfQueuedJobs & " jobs in " & scheduler.Name & _
            " queue. Select Yes to wait for available resources or No to cancel.", 4, "Wolfram Mathematica")

   job.Refresh()

   status = job.State.ToString

   if (msg = 7 and status <> "Running") then

      scheduler.CancelJob job, "Failed to start. Cancel by Mathematica."

      WScript.Quit(-1)

   end if

   if (msg = 7 and status = "Running") then

      msg = MsgBox ("Remote kernel is running.", 0, "Wolfram Mathematica")

   end if

   Do While status = "Queued"

      WScript.Sleep 2000

      job.Refresh()

      status = job.State.ToString

   Loop

end if


if (status <> "Running") then

   scheduler.CancelJob job, "Failed to start. Cancel by Wolfram Mathematica."

   msg = MsgBox ("An error occurs during the start-up of the remote kernel", 0, "Wolfram Mathematica")

   WScript.Quit(-1)

end if
