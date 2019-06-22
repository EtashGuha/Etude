(* ::Package:: *)

(* ::Title:: *)
(*InstallRobotTools*)

(* ::Section:: *)
(*Annotations*)

(* :Title: InstallRobotTools.m *)

(* :Author:
        Brenton Bostick
        brenton@wolfram.com
*)

(* :Package Version: 1.2 *)

(* :Mathematica Version: 7.0 *)

(* :Copyright: RobotTools source code (c) 2005-2008 Wolfram Research, Inc. All rights reserved. *)

(* :Discussion:
   Implementation of InstallRobotTools.
*)

(* ::Section:: *)
(*Information*)

`Information`CVS`$InstallRobotToolsId = "$Id: InstallRobotTools.m,v 1.15 2009/09/08 17:09:25 brenton Exp $"

(* ::Section:: *)
(*Public*)

InstallExternalPackages::usage =
"InstallExternalPackages is an option that tells InstallRobotTools whether or not to install platform-specific packages for\
advanced features."

InstallRobotTools::usage =
"InstallRobotTools[] installs J/Link, loads all necessary Java classes, initializes all necessary variables, and registers all\
periodic functions necessary for using RobotTools."

UninstallRobotTools::usage =
"UninstallRobotTools[] release the robot object and unregisters the periodic function that checks if the front end crashed."

UseRobotFacadeClass::usage =
"UseRobotFacadeClass is an option that tells InstallRobotTools whether or not to use the custom RobotFacade class instead of the\
Robot class."

(* ::Section:: *)
(*Package*)

Begin["`Package`"]

$javaKeys

$javaButtons

$Robot

End[] (*`Package`*)

(* ::Section:: *)
(*Private*)

(* ::Subsection:: *)
(*Begin*)

Begin["`InstallRobotTools`Private`"]

(* ::Subsection:: *)
(*Messages*)

InstallRobotTools::perm =
"The current security manager does not allow robots to be created (much to the relief of Sarah Connor)."

InstallRobotTools::headless =
"A display, keyboard, and mouse cannot be supported in this environment."

(* ::Subsection:: *)
(*InstallRobotTools*)

setupJLink[] :=
	Module[{$javaKeysFileName = ToFileName[{$RobotToolsTextResourcesDirectory}, "JavaKeys.m"],
		$javaButtonsFileName = ToFileName[{$RobotToolsTextResourcesDirectory}, "JavaButtons.m"]}
		,
		InstallJava[];
		(*check if we need to reinstall with Java 9*)
		If[ValueQ[RobotTools`Java9JDK] && FileExistsQ[RobotTools`Java9JDK],
			(*THEN*)
			(*we have to load the java 9 jdk*)
			ReinstallJava[CommandLine->RobotTools`Java9JDK]
		];
		
		LoadJavaClass["java.lang.System"];
		LoadJavaClass["java.awt.AWTPermission"];
		LoadJavaClass["java.awt.GraphicsEnvironment"];
		LoadJavaClass["java.awt.event.KeyEvent"];
		LoadJavaClass["java.awt.event.InputEvent"];
		LoadJavaClass["java.awt.Color"];
		(*check if we have a working Java 9*)
		RobotTools`HavaJava9 = FromDigits[StringSplit[System`getProperty["java.version"], "."|"_"][[1]]] >= 9;

		(* define $javaKeys and $JavaButtons here, after the KeyEvent and InputEvent classes have been loaded, so that the RHSs are
		evaluated to integers, and not kept as unevaluated symbols, which would then just evaluate to integers at runtime, albeit very
		slowly *)
		$javaKeys = Get[$javaKeysFileName];
		$javaButtons = Get[$javaButtonsFileName]
	]

checkIfCreateRobotPermission[] :=
	Module[{sm, perm},
		JavaBlock[
			sm = java`lang`System`getSecurityManager[];
			perm = JavaNew["java.awt.AWTPermission", "createRobot"];
			Check[
				sm@checkPermission[perm]
				,
				Message[InstallRobotTools::perm];
				$Failed
				,
				(* we are "expecting" this message, so don't use throwIfMessage *)
				Java::excptn
			]
		]
	]

checkIfHeadless[] :=
	If[java`awt`GraphicsEnvironment`isHeadless[],
		Message[InstallRobotTools::headless];
		$Failed
	]

createNewRobot[useRobotFacadeClass:boolPat] :=
	If[useRobotFacadeClass,
		AddToClassPath[FileNameJoin[{$RobotToolsDirectory, "Java","RobotTools.jar"}]];
		LoadJavaClass["com.wolfram.robottools.RobotFacade"];
		JavaNew["com.wolfram.robottools.RobotFacade"]
		,
		LoadJavaClass["java.awt.Robot"];
		JavaNew["java.awt.Robot"]
	]

Unprotect[InstallRobotTools]

Options[InstallRobotTools] = {InstallExternalPackages -> False, UseRobotFacadeClass -> False}

InstallRobotTools[OptionsPattern[]] :=
	Module[{installExternalPackages, useRobotFacadeClass},
		{installExternalPackages, useRobotFacadeClass} = OptionValue[{InstallExternalPackages, UseRobotFacadeClass}];
		Catch[
			throwIfMessage[
				(* $Robot is used as the flag to determine if everything has to be installed *)
				If[!JavaObjectQ[$Robot],
					setupJLink[];
					checkIfCreateRobotPermission[];
					checkIfHeadless[];
					$Robot = createNewRobot[useRobotFacadeClass];
					If[installExternalPackages,
						Switch[$InterfaceEnvironment,
							(*"Macintosh",
							RobotTools`InstallAppleScript[]
							,*)
							"Windows",
							Win32`InstallWin32[]
							,
							"X",
							XlibLink`InstallXlibLink[]
							,
							(*HoldPattern[$InterfaceEnvironment]*)
							_,
							Null
						]
					];
					(* make sure that the front end is in front, this also works around a bug in the AppleScript Java libraries *)
					If[$FrontEnd =!= Null,
						FrontEndTokenExecute["BringToFront"]
					]
				]
			]
		]
	]

SetAttributes[InstallRobotTools, ReadProtected]

InstallRobotTools[args___] :=
	(ArgumentCountQ[InstallRobotTools, System`FEDump`NonOptionArgCount[{args}], 0, 0]; $Failed)

SyntaxInformation[InstallRobotTools] = {"ArgumentsPattern" -> {OptionsPattern[]}}

Protect[InstallRobotTools]

(* ::Subsection:: *)
(*UninstallRobotTools*)

Unprotect[UninstallRobotTools]

UninstallRobotTools[] :=
	(
		ReleaseJavaObject[$Robot];
		removePeriodicals[]
	)

SetAttributes[UninstallRobotTools, ReadProtected]

Protect[UninstallRobotTools]

(* ::Subsection:: *)
(*End*)

End[] (*`InstallRobotTools`Private`*)
