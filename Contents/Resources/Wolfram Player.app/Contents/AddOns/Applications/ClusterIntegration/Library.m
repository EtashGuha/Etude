(* :Name: Library.m *)

(* :Title: Library functions *)

(* :Context: ClusterIntegration`Library` *)

(* :Author: Charles Pooh *)

(* :Summary: This package provides internal tools for CIP *)

(* :Copyright: (c) 2006 - 2008 Wolfram Research, Inc. *)

(* :Sources: *)

(* :Package Version: 2.0 *)

(* :Mathematica Version: 7.0 *)

(* :History: *)

(* :Keywords: None *)

(* :Warnings: None *)

(* :Limitations: *)

(* :Discussion: *)

(* :Requirements: *)

(* :Examples: None *)


(*****************************************************************************)


BeginPackage["ClusterIntegration`Library`", "JLink`"]


(* usage *)

RunCommand::usage = "RunCommand[cmd] runs cmd as an external operating \
system command."


Begin["`Private`"]


(* ************************************************************************* **

                               Utility functions

   Comments:

   ToDo:

** ************************************************************************* *)


RunCommand[cmd_String] :=
    Block[{res},
        res = Quiet[JavaBlock[runCommand[cmd]]];
        res /; FreeQ[res, $Failed]
    ]


RunCommand[___] := $Failed


(* :runCommand: *)

runCommand[cmd_String] /; (Needs["JLink`"] === Null) :=
    Block[{res, res1, jlink, java, runtime, process},

        jlink = InstallJava[];
        (
          java = LoadJavaClass["java.lang.Runtime"];
          (
            runtime = Runtime`getRuntime[];
            process = runtime@exec[cmd];
            process@waitFor[];

            res1 = process@exitValue[];
            (
              res = readFromPipe[process];
              res /; FreeQ[res, $Failed]

            ) /; res1 == 0

          ) /; Head[java] === JavaClass

        ) /; Head[jlink] === LinkObject

    ]


runCommand[cmd_String, type_:"Text"] /; (Needs["JLink`"] =!= Null) :=
    Block[{res},
        res = Check[Import["!" <> cmd, type], $Failed];
        res /; FreeQ[res, $Failed] && (Head[res] =!= Import)
    ]


runCommand[___] := $Failed


(* :readFromPipe: *)

readFromPipe[process_] :=
    Block[{res, processStream, bytesAvailable, bytesRead, bytesArray},

        processStream = process@getInputStream[];
        (
          bytesAvailable = processStream@available[];
          (
            bytesArray = JavaNew["[B", bytesAvailable];
            (
              bytesRead = processStream@read[bytesArray, 0, bytesAvailable];
              (
                res = StringJoin @@ (
                 FromCharacterCode /@ JavaObjectToExpression[bytesArray]);

                res /; StringQ[res]

              ) /; (bytesRead == bytesAvailable)

            ) /; Head[bytesArray] === Symbol

          ) /; Positive[bytesAvailable]

        ) /; Head[processStream] === Symbol

    ]


readFromPipe[___] := $Failed


(* ************************************************************************* *)


End[]


EndPackage[]