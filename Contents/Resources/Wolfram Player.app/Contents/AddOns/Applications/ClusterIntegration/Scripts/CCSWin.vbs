'******************************************************************************
'
' Cluster Integration Package for gridMathematica
'
' Author: Charles Pooh
'
' Copyright (c) 2006 - 2008 Wolfram Research, Inc. All rights reserved.
'
' Mathematica version: 7.0
'
' Summary: Microsoft Compute Cluster Server 2003 remote kernel script
'
'******************************************************************************


' Verify Arguments.Count
'

if (WScript.Arguments.Count <> 4 and WScript.Arguments.Count <> 3) then

   msg = MsgBox ("Remote command called with " & WScript.Arguments.Count & _
          " arguments; 3 or 4 arguments expected for Kernel configuration" & _
          " options.", 0, "Wolfram Mathematica")

   WScript.Quit(-1)

end if


' Definition of Shell object
'

dim WshShell

Set WshShell = CreateObject("WScript.Shell")


' Get the parameters
'

Set arguments = WScript.Arguments

mathKernel = Chr(34) & arguments(0) & Chr(34)

if (arguments.Count = 3) then

   kernelOptions = "-mathlink -LinkMode Connect -LinkProtocol TCPIP -LinkName"

   linkName = arguments(1)

   scheduler = arguments(2)

end if

if (arguments.Count = 4) then

   kernelOptions = arguments(1)

   linkName = arguments(2)

   scheduler = arguments(3)

end if


' Connect to the cluster
'

Set computeCluster = CreateObject("Microsoft.ComputeCluster.Cluster")

computeCluster.Connect(scheduler)


' Create and submit job
'

Set job = computeCluster.CreateJob

job.Name = "Wolfram Mathematica - " & Date & " - " & Time

jobID = computeCluster.AddJob((job))


Set task = computeCluster.CreateTask

task.Name = "Mathematica Kernel"

task.CommandLine = mathKernel & " " & kernelOptions & " " & linkName

taskID = computeCluster.AddTask(jobID, (task))


computeCluster.SubmitJob jobID, "", "", False, 0


' Check status of job
'

WScript.Sleep 5000

Set job = computeCluster.GetJob(jobID)

if (job.Status = 1) then

   Set counter = computeCluster.ClusterCounter

   msg = MsgBox (counter.NumberOfQueuedJobs & " jobs in " & computeCluster.Name & _
            " queue. Select Yes to wait for available resources or No to cancel.", 4, "Wolfram Mathematica")

   Set job = computeCluster.GetJob(jobID)

   if (msg = 7 and job.Status <> 2) then

      computeCluster.CancelJob jobID, "Failed to start. Cancel by Wolfram Mathematica."

      WScript.Quit(-1)

   end if

   if (msg = 7 and job.Status = 2) then

      msg = MsgBox ("Remote kernel is running.", 0, "Wolfram Mathematica")

   end if

   Do While job.Status = 1

      WScript.Sleep 2000

      Set job = computeCluster.GetJob(jobID)

   Loop

end if


if (job.Status <> 2) then

   msg = MsgBox ("An error occurs during the start-up of the remote kernel", 0, "Wolfram Mathematica")

   computeCluster.CancelJob jobID, "Failed to start. Cancel by Wolfram Mathematica."

   WScript.Quit(-1)

end if
